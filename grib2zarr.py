"""
grib2zarr.py - Extract GRIB2 messages and store them in Zarr format using asyncio.

This module reads one or more GRIB2 files, extracts parameters described by a
YAML configuration file, and writes the data into a Zarr store.  A
producer/consumer pattern is used with asyncio.Queue to decouple reading
from writing.

Variables are identified by their GRIB2 discipline / parameterCategory /
parameterNumber keys.  Dataset shape, chunk layout, coordinates, CRS and
CF attributes are all derived from the configuration file.

Usage:
    python grib2zarr.py GRIB_FILE [GRIB_FILE ...] --config CONFIG [--output ZARR_PATH]

Example:
    python grib2zarr.py fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \\
        --config config.yaml --output myfile.zarr
"""

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from typing import List, Union

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

# Default output path
DEFAULT_ZARR_PATH = "myfile.zarr"

_log = logging.getLogger(__name__)

# Named constants used in time-index matching
_SECONDS_PER_HOUR: float = 3600.0
_TIME_MATCH_TOLERANCE_HOURS: float = 1e-6


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
    # For datetime64 coordinates xarray encodes to int64 with CF units.
    # zarr v2 defaults to fill_value=0 for int64, so any time step whose
    # encoded value is 0 (e.g. step 0 = reference time when units are
    # "hours since <reference_time>") would be masked as NaT on read-back.
    # Use the largest int64 value as fill_value instead – it is far outside
    # any realistic meteorological time range.
    encoding = {
        name: {"_FillValue": np.iinfo(np.int64).max}
        for name, coord in ds.coords.items()
        if np.issubdtype(coord.dtype, np.int64)
    }
    ds.to_zarr(open_store(zarr_path), mode="w", zarr_format=2, compute=False, encoding=encoding)
    return ds


def _build_var_matcher(config: dict) -> list:
    """Build a list of variable matchers from a parsed config.

    Each entry is a tuple ``(var_name, grib2_keys, dims)`` where *grib2_keys*
    is a dict of the combined GRIB2 identification keys for that variable —
    its own discipline/parameterCategory/parameterNumber keys **plus** any
    ``grib2`` keys declared on its dimension references (e.g.
    ``typeOfFirstFixedSurface`` from the vertical coordinate).  Including the
    vertical-coordinate keys ensures that two variables sharing the same
    parameter numbers but placed on different level types (e.g. hybrid-sigma
    vs. height-above-ground) are matched uniquely.

    Parameters
    ----------
    config:
        Parsed configuration dictionary.

    Returns
    -------
    list of (str, dict, list)
    """
    matchers = []
    for var in config.get("variables", []):
        var_name = var["name"]
        grib2_keys = dict(var.get("grib2", {}))
        dims = var.get("dims", [])
        # Merge grib2 keys from dimension references so that level-type keys
        # (e.g. typeOfFirstFixedSurface) become part of the variable match.
        for dim_ref in dims:
            if isinstance(dim_ref, dict):
                dim_grib2 = dim_ref.get("grib2", {})
                if dim_grib2:
                    grib2_keys.update(dim_grib2)
        matchers.append((var_name, grib2_keys, dims))
    return matchers


def _find_level_index(level: int, msg_keys: dict, dims: list):
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
            coord_values = _eval_values(dim_ref.get("values", []))
            if level in coord_values:
                return coord_values.index(level)
            # Level is outside the configured coordinate range → skip message
            return None
    return None


def _find_time_index(msg_keys: dict, dims: list):
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
        step_values = _eval_values(dim_ref.get("values", []))
        for idx, sv in enumerate(step_values):
            if abs(float(sv) - delta_hours) < _TIME_MATCH_TOLERANCE_HOURS:
                return idx
        # Validity time is outside the configured step range → skip message
        return None
    # No datetime-based time dimension: single-time-step fallback
    return 0


async def read_grib(grib_file_paths: Union[str, List[str]], matchers: list):
    """Async generator yielding matched GRIB messages.

    Each yielded item is a tuple::

        (var_name, t_idx, z_idx, values)

    where *t_idx* is the zero-based index into the time coordinate (derived
    from the message's ``validityDate``/``validityTime`` keys when the time
    dimension carries a ``reference_time``, or ``0`` for single-step configs),
    *z_idx* is the zero-based index into the variable's vertical coordinate,
    and *values* is the flat numpy array of grid values.

    Only messages that match at least one variable in *matchers* **and**
    whose level **and** validity time fall within the configured coordinate
    ranges are yielded.

    GRIB key names used for matching are derived dynamically from the config:
    all keys present in each variable's ``grib2`` dict are checked, so adding
    extra identification keys (e.g. ``typeOfStatisticalProcessing``) to the
    config automatically takes effect without code changes.

    Parameters
    ----------
    grib_file_paths:
        Path (or list of paths) to the GRIB2 file(s) to read.  When a list
        is supplied all files are processed in order as if they were
        concatenated.
    matchers:
        List produced by :func:`_build_var_matcher`.
    """
    if isinstance(grib_file_paths, str):
        grib_file_paths = [grib_file_paths]

    # Pre-compute the union of all GRIB key names that need to be fetched.
    # This covers both variable-identification keys (from each variable's
    # grib2 dict) and dimension-identification keys (from each dim's grib2
    # dict), so a single pass per message is sufficient.
    # validityDate and validityTime are always fetched to support
    # datetime-based time-index matching.
    keys_needed: set = {"validityDate", "validityTime"}
    for _var_name, grib2_keys, dims in matchers:
        keys_needed.update(grib2_keys.keys())
        for dim_ref in dims:
            if isinstance(dim_ref, dict):
                keys_needed.update(dim_ref.get("grib2", {}).keys())

    for grib_file_path in grib_file_paths:
        _log.info("extract  file='%s'  scanning …", grib_file_path)
        t_file_start = time.perf_counter()
        matched_count = 0
        with open(grib_file_path, "rb") as f:
            while True:
                gid = codes_grib_new_from_file(f)
                if gid is None:
                    break

                level = codes_get(gid, "level", ktype=int)

                # Fetch all keys required for matching in one pass.
                # Request values as ``int`` (ktype=int) so that coded keys such as
                # ``typeOfFirstFixedSurface`` are always returned as integers
                # (e.g. 105) rather than string abbreviations (e.g. 'ml').
                # This ensures the values match the integer constants declared in
                # the config YAML.
                msg_keys: dict = {}
                for key in keys_needed:
                    try:
                        msg_keys[key] = codes_get(gid, key, ktype=int)
                    except Exception:
                        try:
                            msg_keys[key] = codes_get(gid, key)
                        except Exception:
                            pass  # Key absent in this message type

                matched = None
                for var_name, grib2_keys, dims in matchers:
                    if all(msg_keys.get(k) == v for k, v in grib2_keys.items()):
                        matched = (var_name, dims)
                        break

                if matched is not None:
                    var_name, dims = matched
                    z_idx = _find_level_index(level, msg_keys, dims)
                    t_idx = _find_time_index(msg_keys, dims)
                    if z_idx is not None and t_idx is not None:
                        values = codes_get_values(gid)
                        if codes_get(gid, "bitmapPresent"):
                            bitmap = np.array(codes_get_array(gid, "bitmap"), dtype=bool)
                            values[~bitmap] = np.nan
                        codes_release(gid)
                        matched_count += 1
                        yield (var_name, t_idx, z_idx, values)
                    else:
                        codes_release(gid)
                else:
                    codes_release(gid)

                # Yield control back to the event loop between messages so that
                # the consumer can make progress while reading is ongoing.
                await asyncio.sleep(0)
        t_file_elapsed = time.perf_counter() - t_file_start
        _log.info(
            "extract  file='%s'  matched=%d  elapsed=%.1fs",
            grib_file_path,
            matched_count,
            t_file_elapsed,
        )


async def write_slice(
    ds: xr.Dataset,
    var_name: str,
    t_idx: int,
    z_idx: int,
    values,
    zarr_path: str,
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
    zarr_path:
        Path of the Zarr store to update.
    """
    da_var = ds[var_name]
    dim_names = list(da_var.dims)

    # Last two dims are always the spatial (y, x) dims
    spatial_dims = set(dim_names[-2:]) if len(dim_names) >= 2 else set()

    # Reshape flat values to (ny, nx) and cast to float32.
    ny = da_var.sizes[dim_names[-2]]
    nx = da_var.sizes[dim_names[-1]]
    grid = values.reshape(ny, nx).astype("float32")

    # Build the numpy index tuple for writing directly into the zarr array.
    # Integer indices are used for the time and vertical dimensions so that
    # only the single target slice is overwritten.
    np_idx = []
    for dim in dim_names:
        if dim in spatial_dims:
            np_idx.append(slice(None))
        elif dim in ds.coords and ds.coords[dim].attrs.get("axis") == "T":
            np_idx.append(t_idx)
        else:
            np_idx.append(z_idx)

    # Write directly to the Zarr store.  The dataset data variables are
    # backed by lazy dask arrays, so modifying ``ds[var_name].values`` only
    # updates a temporary computed copy and never persists.  Opening the store
    # with zarr and indexing directly avoids that problem.
    store = zarr.open(open_store(zarr_path), mode="r+")
    store[var_name][tuple(np_idx)] = grid


async def producer(
    queue: asyncio.Queue,
    grib_file_paths: Union[str, List[str]],
    matchers: list,
) -> None:
    """Read GRIB messages and push them onto *queue*.

    A sentinel value of ``None`` is placed on the queue after all messages
    have been read to signal the consumer to stop.

    Parameters
    ----------
    queue:
        Shared asyncio queue.
    grib_file_paths:
        Path (or list of paths) to the GRIB2 file(s).
    matchers:
        List produced by :func:`_build_var_matcher`.
    """
    async for message in read_grib(grib_file_paths, matchers):
        await queue.put(message)
    await queue.put(None)  # Sentinel – no more messages


async def consumer(
    queue: asyncio.Queue,
    ds: xr.Dataset,
    zarr_path: str,
) -> None:
    """Consume messages from *queue* and write them to the Zarr store.

    Parameters
    ----------
    queue:
        Shared asyncio queue populated by :func:`producer`.
    ds:
        In-memory dataset used for slice-wise Zarr writes.
    zarr_path:
        Path of the Zarr store to update.
    """
    while True:
        message = await queue.get()
        if message is None:
            queue.task_done()
            break
        var_name, t_idx, z_idx, values = message
        await write_slice(ds, var_name, t_idx, z_idx, values, zarr_path)
        queue.task_done()


async def main(
    grib_file_paths: Union[str, List[str]],
    zarr_path: str,
    config_path: str,
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
    """
    config = load_config(config_path)
    ds = initialise_zarr(zarr_path, config)
    matchers = _build_var_matcher(config)
    var_names = [m[0] for m in matchers]

    if isinstance(grib_file_paths, list):
        n_files = len(grib_file_paths)
    else:
        n_files = 1

    _log.info(
        "extract  files=%d  variables=%s  output='%s'",
        n_files,
        var_names,
        zarr_path,
    )
    t_extract_start = time.perf_counter()

    queue: asyncio.Queue = asyncio.Queue()
    producer_task = asyncio.create_task(producer(queue, grib_file_paths, matchers))
    consumer_task = asyncio.create_task(consumer(queue, ds, zarr_path))

    await asyncio.gather(producer_task)
    await queue.join()
    await consumer_task

    t_extract_elapsed = time.perf_counter() - t_extract_start
    _log.info(
        "extract  done  total=%.1fs  output='%s'",
        t_extract_elapsed,
        zarr_path,
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
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging (shows progress, timing, etc.).",
    )
    return parser.parse_args(argv)


def cli() -> None:
    """Console-script entry point installed by ``pip install``."""
    args = _parse_args(sys.argv[1:])
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    asyncio.run(main(args.grib_files, args.zarr_path, args.config))


if __name__ == "__main__":
    cli()

