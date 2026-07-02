"""
grib2zarr.py - Extract GRIB2 messages and store them in Zarr format.

This module reads one or more GRIB2 files, extracts parameters described by a
YAML configuration file, and writes the data into a Zarr store.  A
producer/consumer pattern is used to decouple reading from writing.

Two pipelines are provided:

* **Serial** (default) – a single ``asyncio`` producer/consumer, unchanged from
  earlier versions of the tool.  Used when ``--jobs 1 --writers 1``.
* **Parallel** – a bounded :class:`queue.Queue` with N reader threads (one per
  input file) and M writer threads writing disjoint slabs to the Zarr store
  concurrently.  Selected via ``--jobs`` / ``--writers`` greater than 1.

Variables are identified by their GRIB2 discipline / parameterCategory /
parameterNumber keys.  Dataset shape, chunk layout, coordinates, CRS and
CF attributes are all derived from the configuration file.

Usage:
    python grib2zarr.py GRIB_FILE [GRIB_FILE ...] --config CONFIG \\
        [--output ZARR_PATH] [--jobs N] [--writers M]

Example:
    python grib2zarr.py fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \\
        --config config.yaml --output myfile.zarr --jobs 4 --writers 4
"""

import argparse
import asyncio
import logging
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import xarray as xr
import zarr
from eccodes import (
    codes_get,
    codes_get_array,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
)

from config_parser import build_dataset, load_config, _eval_values
from s3_store import open_store

_DBG = logging.DEBUG  # local alias avoids repeated attribute lookup in hot paths

# Default output path
DEFAULT_ZARR_PATH = "myfile.zarr"

_log = logging.getLogger(__name__)

# Named constants used in time-index matching
_SECONDS_PER_HOUR: float = 3600.0
_TIME_MATCH_TOLERANCE_HOURS: float = 1e-6

# Sentinel placed on the parallel queue to terminate writer threads.
_STOP = object()

logger = logging.getLogger(__name__)


def initialise_zarr(zarr_path: str, config: dict) -> xr.Dataset:
    """Create an empty Zarr store from a YAML dataset configuration.

    Shape, chunk layout, coordinates, grid-mapping variables and CF attributes
    are all derived from *config* via :func:`config_parser.build_dataset`.

    Parameters
    ----------
    zarr_path:
        File-system path for the output Zarr store.
    config:
        Parsed configuration dictionary (from :func:`config_parser.load_config`).

    Returns
    -------
    xr.Dataset
        The in-memory dataset whose Zarr store was just initialised.
    """
    ds = build_dataset(config)
    # zarr v2 defaults to fill_value=0 / 0.0 for numeric arrays.  xarray
    # masks values equal to _FillValue as NaN/NaT when reading back with CF
    # decoding, so any coordinate whose first value is 0 (e.g. x[0]=0,
    # time step 0 = reference time encoded as int64 0) would become NaN/NaT.
    #
    # Fix: explicitly set _FillValue in the zarr encoding for every coordinate:
    #   • datetime64 → xarray encodes as int64; use INT64_MAX as sentinel.
    #   • float64    → use NaN so that 0.0 is never treated as missing.
    encoding: dict = {}
    for name, coord in ds.coords.items():
        if np.issubdtype(coord.dtype, np.datetime64):
            encoding[name] = {"_FillValue": np.iinfo(np.int64).max}
        elif np.issubdtype(coord.dtype, np.floating):
            entry: dict = {"_FillValue": np.nan}
            # 2-D coordinate arrays (e.g. latitude, longitude) are stored as a
            # single chunk so that the entire grid is read in one I/O request.
            if coord.ndim == 2:
                entry["chunks"] = coord.shape
            encoding[name] = entry
    ds.to_zarr(open_store(zarr_path), mode="w", zarr_format=2, compute=False, encoding=encoding)
    return ds


def _build_var_matcher(config: dict) -> list:
    """Build a list of variable matchers from a parsed config.

    Each entry is a tuple ``(var_name, grib2_keys, dims, valid_levels)`` where
    *grib2_keys* is a dict of the combined GRIB2 identification keys for that
    variable — its own discipline/parameterCategory/parameterNumber keys
    **plus** any ``grib2`` keys declared on its dimension references (e.g.
    ``typeOfFirstFixedSurface`` from the vertical coordinate).  Including the
    vertical-coordinate keys ensures that two variables sharing the same
    parameter numbers but placed on different level types (e.g. hybrid-sigma
    vs. height-above-ground) are matched uniquely.

    *valid_levels* is the set of GRIB ``level`` values that are valid for the
    variable's vertical dimension (derived from that dimension's ``values``
    list).  When two variables share all ``grib2`` keys but differ only by
    level value (e.g. temperature at height 0 m vs. 2 m), this set is used
    to disambiguate them during matching.  ``None`` means no level-based
    filtering is applied (e.g. for variables with no vertical dimension).

    Parameters
    ----------
    config:
        Parsed configuration dictionary.

    Returns
    -------
    list of (str, dict, list, set or None)
    """
    matchers = []
    for var in config.get("variables", []):
        var_name = var["name"]
        grib2_keys = dict(var.get("grib2", {}))
        dims = var.get("dims", [])
        valid_levels = None
        # Merge grib2 keys from dimension references so that level-type keys
        # (e.g. typeOfFirstFixedSurface) become part of the variable match.
        # Also collect the valid level values from the vertical dimension so
        # that variables with the same level-type key but different level
        # values (e.g. height0=0 m vs. height2=2 m) can be distinguished.
        for dim_ref in dims:
            if isinstance(dim_ref, dict):
                dim_grib2 = dim_ref.get("grib2", {})
                if dim_grib2:
                    grib2_keys.update(dim_grib2)
                    coord_values = _eval_values(dim_ref.get("values", []))
                    if coord_values:
                        valid_levels = set(coord_values)
        matchers.append((var_name, grib2_keys, dims, valid_levels))
    # Sort most-specific matchers first (most grib2 keys) so that a variable
    # whose keys are a strict superset of another's is always checked first.
    # This prevents, e.g., a 3-key matcher from swallowing messages that
    # belong to a 4-key matcher with the same 3 keys plus an extra one such
    # as typeOfStatisticalProcessing.
    matchers.sort(key=lambda m: len(m[1]), reverse=True)
    return matchers


def _find_level_index(level: int, msg_keys: dict, dims: list, _coord_cache: dict):
    """Return the index of *level* within the matching vertical coordinate.

    Searches *dims* for a dimension whose ``grib2`` keys all match the
    corresponding values in *msg_keys*, then looks up *level* in that
    dimension's configured coordinate values.

    Returns the zero-based list index when the level is found in the
    configured coordinate values, or ``None`` if no dimension matches or
    the level value falls outside the configured coordinate range.

    Parameters
    ----------
    level:
        GRIB ``level`` key value for the current message.
    msg_keys:
        Dict of GRIB key/value pairs already fetched from the current message.
    dims:
        List of dimension reference dicts from the variable configuration.
    _coord_cache:
        Mutable dict used to cache evaluated coordinate lists so that
        ``_eval_values`` is not called more than once per unique *dim_ref*
        object.  Pass the same dict across all calls to share the cache.

    Returns
    -------
    int or None
        Zero-based index into the coordinate array, or ``None`` if the
        level is not present in the configured coordinate values.
    """
    for dim_ref in dims:
        if not isinstance(dim_ref, dict):
            continue
        grib2 = dim_ref.get("grib2", {})
        if not grib2:
            continue
        # All grib2 keys declared for this dim must match the current message
        if all(msg_keys.get(k) == v for k, v in grib2.items()):
            dim_id = id(dim_ref)
            if dim_id not in _coord_cache:
                _coord_cache[dim_id] = _eval_values(dim_ref.get("values", []))
            coord_values = _coord_cache[dim_id]
            if level in coord_values:
                return coord_values.index(level)
            # Level is outside the configured coordinate range → skip message
            return None
    return None


def _find_time_index(msg_keys: dict, dims: list, _coord_cache: dict):
    """Return the time-dimension index for the current GRIB2 message.

    Searches *dims* for a dimension that carries a ``reference_time`` field
    (marking it as the time/forecast-period axis).  The validity datetime of
    the message is reconstructed from the ``validityDate`` (YYYYMMDD integer)
    and ``validityTime`` (HHMM integer) keys in *msg_keys* and compared
    against the configured step offsets (hours from ``reference_time``).

    When no dimension with ``reference_time`` is found the function returns
    ``0`` so that single-time-step configurations continue to work without
    modification.

    Parameters
    ----------
    msg_keys:
        Dict of GRIB key/value pairs already fetched from the current message.
    dims:
        List of dimension reference dicts from the variable configuration.
    _coord_cache:
        Mutable dict used to cache evaluated step-value lists so that
        ``_eval_values`` is not called more than once per unique *dim_ref*
        object.  Pass the same dict across all calls to share the cache.

    Returns
    -------
    int or None
        Zero-based index into the time coordinate array, ``0`` when no
        datetime-based time dimension is configured, or ``None`` when the
        message validity time falls outside the configured step range.
    """
    for dim_ref in dims:
        if not isinstance(dim_ref, dict):
            continue
        ref_time_str = dim_ref.get("reference_time")
        if ref_time_str is None:
            continue
        # This dimension is the datetime-based time axis.
        validity_date = msg_keys.get("validityDate")
        validity_time = msg_keys.get("validityTime")
        if validity_date is None or validity_time is None:
            return None
        ref_dt = datetime.fromisoformat(ref_time_str)
        valid_dt = datetime(
            validity_date // 10000,
            (validity_date % 10000) // 100,
            validity_date % 100,
            validity_time // 100,
            validity_time % 100,
        )
        delta_hours = (valid_dt - ref_dt).total_seconds() / _SECONDS_PER_HOUR
        dim_id = id(dim_ref)
        if dim_id not in _coord_cache:
            _coord_cache[dim_id] = _eval_values(dim_ref.get("values", []))
        step_values = _coord_cache[dim_id]
        for idx, sv in enumerate(step_values):
            if abs(float(sv) - delta_hours) < _TIME_MATCH_TOLERANCE_HOURS:
                return idx
        # Validity time is outside the configured step range → skip message
        return None
    # No datetime-based time dimension: single-time-step fallback
    return 0


def _keys_needed_for(matchers: list) -> set:
    """Return the union of GRIB key names required to identify a message.

    Includes ``validityDate`` / ``validityTime`` (always needed for
    datetime-based time-index matching) plus every key referenced by any
    variable's ``grib2`` dict or by any of its dimension references.
    """
    keys_needed: set = {"validityDate", "validityTime"}
    for _var_name, grib2_keys, dims, _valid_levels in matchers:
        keys_needed.update(grib2_keys.keys())
        for dim_ref in dims:
            if isinstance(dim_ref, dict):
                keys_needed.update(dim_ref.get("grib2", {}).keys())
    return keys_needed



def _fetch_msg_keys(gid, keys_needed: set) -> dict:
    """Fetch every key in *keys_needed* from GRIB message *gid* in one pass.

    Values are requested as ``int`` (``ktype=int``) so that coded keys such as
    ``typeOfFirstFixedSurface`` are always returned as integers (e.g. 105)
    rather than string abbreviations (e.g. ``'ml'``).  This ensures the values
    match the integer constants declared in the config YAML.
    """
    msg_keys: dict = {}
    for key in keys_needed:
        try:
            msg_keys[key] = codes_get(gid, key, ktype=int)
        except Exception:
            try:
                msg_keys[key] = codes_get(gid, key)
            except Exception:
                pass  # Key absent in this message type
    return msg_keys


def _read_grib_file(
    grib_file_path: str,
    matchers: list,
    coord_cache: Optional[dict] = None,
) -> Iterator[Tuple[str, int, int, "object"]]:
    """Synchronously iterate matched GRIB messages in a single file.

    Each yielded item is a tuple ``(var_name, t_idx, z_idx, values)`` – see
    :func:`read_grib` for the exact semantics.  This function is the
    thread-friendly building block used by both the async serial pipeline
    (via :func:`read_grib`) and the parallel pipeline (executed inside a
    :class:`~concurrent.futures.ThreadPoolExecutor` worker).

    Parameters
    ----------
    grib_file_path:
        Path to a single GRIB2 file.
    matchers:
        List produced by :func:`_build_var_matcher`.
    coord_cache:
        Optional shared cache for coordinate value lists, keyed by
        ``id(dim_ref)``.  When ``None`` a new cache is created for this
        call.  Passing a shared cache across files avoids re-evaluating
        the same ``dim_ref`` dictionaries repeatedly.
    """
    keys_needed = _keys_needed_for(matchers)

    if coord_cache is None:
        coord_cache = {}

    _log.debug(
        "read_grib  file='%s'  matchers=%d  keys_needed=%d  keys=%s",
        grib_file_path,
        len(matchers),
        len(keys_needed),
        sorted(keys_needed),
    )

    _log.info("extract  file='%s'  scanning …", grib_file_path)
    t_file_start = time.perf_counter()
    matched_count = 0
    scanned_count = 0
    # Per-variable match counters for DEBUG diagnostics.
    _var_match_counts: dict = {m[0]: 0 for m in matchers}

    with open(grib_file_path, "rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break
            try:
                scanned_count += 1
                level = codes_get(gid, "level", ktype=int)
                msg_keys = _fetch_msg_keys(gid, keys_needed)

                matched = None
                for var_name, grib2_keys, dims, valid_levels in matchers:
                    if all(msg_keys.get(k) == v for k, v in grib2_keys.items()):
                        if valid_levels is None or level in valid_levels:
                            matched = (var_name, dims)
                            break

                if matched is None:
                    continue

                var_name, dims = matched
                z_idx = _find_level_index(level, msg_keys, dims, coord_cache)
                t_idx = _find_time_index(msg_keys, dims, coord_cache)
                if z_idx is None or t_idx is None:
                    continue

                values = codes_get_values(gid)
                if codes_get(gid, "bitmapPresent"):
                    bitmap = np.array(
                        codes_get_array(gid, "bitmap", ktype=int), dtype=bool
                    )
                    values[~bitmap] = np.nan
                matched_count += 1
                _var_match_counts[var_name] += 1
                yield (var_name, t_idx, z_idx, values)
            finally:
                codes_release(gid)

    t_file_elapsed = time.perf_counter() - t_file_start
    _log.info(
        "extract  file='%s'  scanned=%d  matched=%d  elapsed=%.1fs",
        grib_file_path,
        scanned_count,
        matched_count,
        t_file_elapsed,
    )
    if _log.isEnabledFor(_DBG):
        for vname, cnt in _var_match_counts.items():
            _log.debug(
                "extract  file='%s'  var='%s'  matched=%d",
                grib_file_path,
                vname,
                cnt,
            )


async def read_grib(grib_file_paths: Union[str, List[str]], matchers: list):
    """Async generator yielding matched GRIB messages.

    Thin async wrapper around :func:`_read_grib_file` that iterates the files
    serially and yields to the event loop between messages so the serial
    consumer can make progress while reading is ongoing.
    """
    if isinstance(grib_file_paths, str):
        grib_file_paths = [grib_file_paths]

    # Shared cache for coordinate value lists across all calls to
    # _find_level_index and _find_time_index.  Keyed by id(dim_ref) so that
    # the same dim_ref dict is only evaluated once, regardless of how many
    # messages reference it.
    coord_cache: dict = {}

    for grib_file_path in grib_file_paths:
        for item in _read_grib_file(grib_file_path, matchers, coord_cache):
            yield item
            # Cooperative yield so the consumer task can make progress.
            await asyncio.sleep(0)


def _compute_np_idx(ds: xr.Dataset, var_name: str, t_idx: int, z_idx: int):
    """Return the numpy index tuple for the (t, z) slab of *var_name*.

    The last two dimensions of the variable are treated as the spatial (y, x)
    axes and are written in full; the CF ``axis: T`` dimension is written at
    *t_idx*; any remaining non-spatial dimension is written at *z_idx*.
    """
    da_var = ds[var_name]
    dim_names = list(da_var.dims)
    spatial_dims = set(dim_names[-2:]) if len(dim_names) >= 2 else set()

    np_idx = []
    for dim in dim_names:
        if dim in spatial_dims:
            np_idx.append(slice(None))
        elif dim in ds.coords and ds.coords[dim].attrs.get("axis") == "T":
            np_idx.append(t_idx)
        else:
            np_idx.append(z_idx)
    return tuple(np_idx), dim_names


def _write_slice_sync(
    ds: xr.Dataset,
    store,
    var_name: str,
    t_idx: int,
    z_idx: int,
    values,
) -> None:
    """Write a single level slice into an already-open Zarr store.

    Splitting the actual write from the store-open lets the parallel writers
    share one :func:`zarr.open` handle instead of reopening on every message,
    which is materially cheaper on directory stores and object stores alike.
    """
    da_var = ds[var_name]
    dim_names = list(da_var.dims)
    ny = da_var.sizes[dim_names[-2]]
    nx = da_var.sizes[dim_names[-1]]
    # Cast to float32 so that the on-disk representation matches the dataset
    # dtype produced by :func:`initialise_zarr`.
    grid = values.reshape(ny, nx).astype("float32")

    np_idx, _ = _compute_np_idx(ds, var_name, t_idx, z_idx)
    t_write_start = time.perf_counter()
    store[var_name][np_idx] = grid
    _log.debug(
        "write_slice  var='%s'  t_idx=%d  z_idx=%d  elapsed=%.4fs",
        var_name,
        t_idx,
        z_idx,
        time.perf_counter() - t_write_start,
    )


async def write_slice(
    ds: xr.Dataset,
    var_name: str,
    t_idx: int,
    z_idx: int,
    values,
    store: zarr.Group,
) -> None:
    """Write a single level slice for a variable into the Zarr store.

    The region written is determined by the variable's dimension names stored
    in *ds*.  All spatial dimensions (y, x – the last two dims) are written in
    full; the time dimension (identified by CF ``axis: T``) and the vertical
    dimension are written at the indices given by *t_idx* and *z_idx*
    respectively.

    Parameters
    ----------
    ds:
        In-memory dataset produced by :func:`initialise_zarr`.
    var_name:
        Name of the target data variable in *ds*.
    t_idx:
        Index along the time dimension.
    z_idx:
        Index along the vertical dimension.
    values:
        Flat numpy array of grid values for this message.
    store:
        Already-open Zarr group for the output store.  Opening the store
        once in the caller and reusing it here avoids per-slice filesystem
        overhead that becomes significant with many variables.
    """
    _write_slice_sync(ds, store, var_name, t_idx, z_idx, values)


async def producer(
    queue_: asyncio.Queue,
    grib_file_paths: Union[str, List[str]],
    matchers: list,
) -> None:
    """Read GRIB messages and push them onto *queue_* (serial pipeline)."""
    async for message in read_grib(grib_file_paths, matchers):
        await queue_.put(message)
    await queue_.put(None)  # Sentinel – no more messages


async def consumer(
    queue_: asyncio.Queue,
    ds: xr.Dataset,
    zarr_path: str,
) -> None:
    """Consume messages from *queue_* and write them to the Zarr store.

    The Zarr store is opened **once** at the start and reused for every
    slice write.  Re-opening the store on each write was the dominant IO
    bottleneck when many variables are present: with N variables ×
    T timesteps × C levels the repeated metadata reads scaled linearly with
    the total number of slices written.

    Parameters
    ----------
    queue_:
        Shared asyncio queue populated by :func:`producer`.
    ds:
        In-memory dataset used for slice-wise Zarr writes.
    zarr_path:
        Path of the Zarr store to update.
    """
    # Open the Zarr store once and reuse it for all writes.
    _t_open = time.perf_counter()
    store = zarr.open(open_store(zarr_path), mode="r+")
    _log.debug(
        "consumer  store_open  path='%s'  elapsed=%.4fs",
        zarr_path,
        time.perf_counter() - _t_open,
    )

    _write_count = 0
    _t_write_total = 0.0
    # Per-variable write counters for DEBUG diagnostics.
    _var_write_counts: dict = {}

    while True:
        message = await queue_.get()
        if message is None:
            queue_.task_done()
            break
        var_name, t_idx, z_idx, values = message
        _t0 = time.perf_counter()
        await write_slice(ds, var_name, t_idx, z_idx, values, store)
        _elapsed = time.perf_counter() - _t0
        _write_count += 1
        _t_write_total += _elapsed
        _var_write_counts[var_name] = _var_write_counts.get(var_name, 0) + 1
        queue_.task_done()

    _log.debug(
        "consumer  done  writes=%d  write_time=%.1fs  avg_write=%.4fs",
        _write_count,
        _t_write_total,
        _t_write_total / max(_write_count, 1),
    )
    if _log.isEnabledFor(_DBG):
        for vname, cnt in sorted(_var_write_counts.items()):
            _log.debug("consumer  var='%s'  writes=%d", vname, cnt)


def _check_chunk_alignment(ds: xr.Dataset, var_names: list) -> bool:
    """Return True when every non-spatial dim of every variable has chunk=1.

    When this holds, each ``(var_name, t_idx, z_idx, :, :)`` slab written by
    the pipeline lands on a distinct Zarr chunk file, so writer threads can
    write concurrently without racing on shared chunks.  When it does not,
    the caller must fall back to a single writer to avoid last-writer-wins
    on read-modify-write of shared chunks.
    """
    for var_name in var_names:
        if var_name not in ds.data_vars:
            continue
        da_var = ds[var_name]
        dim_names = list(da_var.dims)
        if len(dim_names) < 2:
            continue
        non_spatial_dims = dim_names[:-2]
        # ``chunks`` on a dask-backed DataArray is a tuple-per-dim of
        # tuple-of-chunk-sizes, aligned with ``dims``.
        chunks = da_var.chunks
        if chunks is None:
            continue
        for dim, dim_chunks in zip(dim_names, chunks):
            if dim not in non_spatial_dims:
                continue
            if any(cs != 1 for cs in dim_chunks):
                return False
    return True


def _run_parallel(
    ds: xr.Dataset,
    zarr_path: str,
    grib_file_paths: List[str],
    matchers: list,
    jobs: int,
    writers: int,
) -> None:
    """Run the parallel reader/writer pipeline.

    Spawns ``min(jobs, len(files))`` reader threads and ``writers`` writer
    threads around a single bounded :class:`queue.Queue`.  Exceptions in any
    worker propagate to the caller once all work has been drained.

    The queue is bounded (``4 * writers``) to cap peak memory: readers block
    once that many decoded slabs are outstanding, applying natural
    back-pressure when writers are the bottleneck.
    """
    n_readers = min(jobs, len(grib_file_paths))
    q: queue.Queue = queue.Queue(maxsize=max(4 * writers, 2 * n_readers))
    reader_errors: List[BaseException] = []
    writer_errors: List[BaseException] = []
    errors_lock = threading.Lock()

    def reader(path: str) -> None:
        try:
            # Each reader gets its own coord cache; the cache is not shared
            # across threads to keep _read_grib_file free of locks.
            for item in _read_grib_file(path, matchers):
                q.put(item)
        except BaseException as exc:  # noqa: BLE001 - surface all failures
            with errors_lock:
                reader_errors.append(exc)
            logger.exception("Reader failed for %s", path)

    def writer() -> None:
        # Each writer opens the store once; disjoint chunk writes are safe.
        store = zarr.open(open_store(zarr_path), mode="r+")
        try:
            while True:
                item = q.get()
                try:
                    if item is _STOP:
                        return
                    var_name, t_idx, z_idx, values = item
                    _write_slice_sync(
                        ds, store, var_name, t_idx, z_idx, values
                    )
                finally:
                    q.task_done()
        except BaseException as exc:  # noqa: BLE001 - surface all failures
            with errors_lock:
                writer_errors.append(exc)
            logger.exception("Writer failed")

    with ThreadPoolExecutor(
        max_workers=n_readers + writers,
        thread_name_prefix="grib2zarr",
    ) as pool:
        writer_futures = [pool.submit(writer) for _ in range(writers)]
        reader_futures = [
            pool.submit(reader, path) for path in grib_file_paths
        ]

        # Wait for all readers to finish producing.
        for f in reader_futures:
            f.result()

        # Signal writers to stop – one sentinel per writer so each one wakes.
        for _ in range(writers):
            q.put(_STOP)

        for f in writer_futures:
            f.result()

    if reader_errors or writer_errors:
        raise reader_errors[0] if reader_errors else writer_errors[0]


async def main(
    grib_file_paths: Union[str, List[str]],
    zarr_path: str,
    config_path: str,
    jobs: int = 1,
    writers: int = 1,
) -> None:
    """Orchestrate the producer/consumer pipeline.

    Parameters
    ----------
    grib_file_paths:
        Path (or list of paths) to the GRIB2 file(s) to read.
    zarr_path:
        Destination Zarr store path.
    config_path:
        Path to the YAML configuration file describing the target dataset.
    jobs:
        Number of parallel reader threads.  ``1`` (default) preserves the
        original serial ``asyncio`` pipeline.
    writers:
        Number of parallel writer threads.  ``1`` (default) preserves the
        original serial ``asyncio`` pipeline.  When the config's non-spatial
        chunk sizes are not all ``1``, this is silently capped at ``1`` to
        avoid racing on shared chunks.
    """
    config = load_config(config_path)
    ds = initialise_zarr(zarr_path, config)
    matchers = _build_var_matcher(config)
    var_names = [m[0] for m in matchers]

    if isinstance(grib_file_paths, str):
        file_list = [grib_file_paths]
    else:
        file_list = list(grib_file_paths)

    _log.info(
        "extract  files=%d  variables=%s  output='%s'",
        len(file_list),
        var_names,
        zarr_path,
    )
    t_extract_start = time.perf_counter()

    use_parallel = (jobs > 1 or writers > 1) and len(file_list) > 0

    if use_parallel and writers > 1:
        if not _check_chunk_alignment(ds, var_names):
            logger.warning(
                "Non-spatial chunk sizes are not all 1; "
                "falling back to a single writer to avoid chunk write races."
            )
            writers = 1

    if use_parallel:
        # The parallel pipeline is thread-based; run it off the event loop
        # so that ``asyncio.run(main(...))`` continues to work unchanged.
        await asyncio.to_thread(
            _run_parallel,
            ds,
            zarr_path,
            file_list,
            matchers,
            jobs,
            writers,
        )
    else:
        queue_: asyncio.Queue = asyncio.Queue()
        producer_task = asyncio.create_task(
            producer(queue_, file_list, matchers)
        )
        consumer_task = asyncio.create_task(consumer(queue_, ds, zarr_path))

        await asyncio.gather(producer_task)
        await queue_.join()
        await consumer_task

    t_extract_elapsed = time.perf_counter() - t_extract_start
    _log.info(
        "extract  done  total=%.1fs  output='%s'",
        t_extract_elapsed,
        zarr_path,
    )

    files_str = ", ".join(f"'{p}'" for p in file_list)
    print(
        f"Done. Variables {', '.join(var_names)} written to '{zarr_path}' "
        f"from {files_str} using config '{config_path}' "
        f"(jobs={jobs}, writers={writers})."
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract parameters from one or more GRIB2 files and store them in Zarr."
        )
    )
    parser.add_argument(
        "grib_files",
        nargs="+",
        metavar="GRIB_FILE",
        help="Path(s) to one or more input GRIB2 files.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_ZARR_PATH,
        metavar="ZARR_PATH",
        dest="zarr_path",
        help=f"Output Zarr store path (default: {DEFAULT_ZARR_PATH}).",
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="CONFIG",
        help=(
            "Path to a YAML configuration file describing the target dataset. "
            "Variables are matched by GRIB2 keys "
            "(discipline/parameterCategory/parameterNumber)."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of parallel reader threads (one per input file). "
            "Default 1 preserves the serial asyncio pipeline."
        ),
    )
    parser.add_argument(
        "--writers",
        type=int,
        default=None,
        metavar="M",
        help=(
            "Number of parallel writer threads. Defaults to --jobs. "
            "Silently capped at 1 when non-spatial chunk sizes are not all 1."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging (shows progress, timing, etc.).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help=(
            "Enable DEBUG-level logging (shows per-slice write timings, "
            "per-variable match counts and IO diagnostics). Implies --verbose."
        ),
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be >= 1")
    if args.writers is None:
        args.writers = args.jobs
    if args.writers < 1:
        parser.error("--writers must be >= 1")
    return args


def cli() -> None:
    """Console-script entry point installed by ``pip install``."""
    args = _parse_args(sys.argv[1:])
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    elif args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    asyncio.run(
        main(
            args.grib_files,
            args.zarr_path,
            args.config,
            jobs=args.jobs,
            writers=args.writers,
        )
    )


if __name__ == "__main__":
    cli()
