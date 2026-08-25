"""
rechunk.py - Single-pass rechunking for Zarr stores.

Implements a single-pass rechunk that reads each source chunk exactly once and
writes each destination chunk exactly once, with no intermediate temporary store.

Algorithm
---------
For each time block ``[t0:t1]`` and each vertical block ``[c0:c1]``:

1. Read all source chunks for this ``(time_block, level_block)`` into a
   contiguous in-memory buffer of shape ``(tlen, clen, Y, X)``.
2. Write the buffer to the destination store in aligned spatial tiles
   ``(t_chunk, c_chunk, spatial_chunk, spatial_chunk)`` so that each
   destination chunk is written exactly once.

Typical RAM usage for float32 data of shape ``(67, 66, 1069, 949)``
with default settings (``c_chunk=None`` → full C axis):

* Buffer: ~6.4 GiB  (24 × 66 × 1069 × 949 elements)

When memory is constrained, set ``--c-chunk`` to a smaller value; the buffer
size is proportional to ``t_chunk × c_chunk × Y × X``.

Usage
-----
As a library::

    from rechunk import rechunk_zarr

    rechunk_zarr(
        src_path="myfile.zarr",
        dst_path="myfile_rechunked.zarr",
        t_chunk=24,
        c_chunk=None,      # defaults to full C axis
        spatial_chunk=100,
    )

    # Zarr v3 output with sharding
    rechunk_zarr(
        src_path="myfile.zarr",
        dst_path="myfile_rechunked_v3.zarr",
        t_chunk=24,
        c_chunk=8,
        spatial_chunk=100,
        zarr_format=3,
        t_shard=48,
        spatial_shard=500,
    )

Via the ``rechunk2zarr`` CLI::

    rechunk2zarr myfile.zarr rechunked.zarr --t-chunk 24 --spatial-chunk 100 --verbose
    rechunk2zarr myfile.zarr rechunked_v3.zarr --zarr-format 3 \
        --t-chunk 24 --t-shard 48 --spatial-chunk 100 --spatial-shard 500
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import sys
import time
import warnings
from typing import Optional

import zarr

from s3_store import open_store

_log = logging.getLogger(__name__)


def _validate_shards(
    zarr_format: int,
    t_chunk: int,
    c_chunk: Optional[int],
    spatial_chunk: int,
    t_shard: Optional[int],
    c_shard: Optional[int],
    spatial_shard: Optional[int],
) -> None:
    """Validate ``zarr_format`` and per-dimension shard sizes.

    Raises :class:`ValueError` when the requested combination is invalid.
    """
    if zarr_format not in (2, 3):
        raise ValueError(
            f"zarr_format must be 2 or 3, got {zarr_format!r}."
        )

    shards = {"t_shard": t_shard, "c_shard": c_shard, "spatial_shard": spatial_shard}
    any_shard = any(v is not None for v in shards.values())
    if any_shard and zarr_format != 3:
        raise ValueError(
            "Sharding (t_shard/c_shard/spatial_shard) requires zarr_format=3."
        )

    checks = [
        ("t_shard", t_shard, "t_chunk", t_chunk),
        ("c_shard", c_shard, "c_chunk", c_chunk),
        ("spatial_shard", spatial_shard, "spatial_chunk", spatial_chunk),
    ]
    for shard_name, shard, chunk_name, chunk in checks:
        if shard is None:
            continue
        if shard <= 0:
            raise ValueError(f"{shard_name} must be a positive integer, got {shard}.")
        if chunk is None:
            raise ValueError(
                f"{shard_name} was specified but {chunk_name} is not set; "
                f"sharding requires an explicit {chunk_name}."
            )
        if shard % chunk != 0:
            raise ValueError(
                f"{shard_name} ({shard}) must be a positive integer multiple "
                f"of {chunk_name} ({chunk})."
            )


def rechunk_zarr(
    src_path: str,
    dst_path: str,
    t_chunk: int = 24,
    c_chunk: Optional[int] = None,
    spatial_chunk: int = 100,
    workers: int = 1,
    zarr_format: int = 2,
    t_shard: Optional[int] = None,
    c_shard: Optional[int] = None,
    spatial_shard: Optional[int] = None,
) -> None:
    """Rechunk a 4-D Zarr array via a single-pass algorithm.

    The source array is expected to have shape ``(T, C, Y, X)`` – typically
    ``(n_timesteps, n_levels, n_lats, n_lons)``.  Only 4-D data arrays are
    rechunked; coordinate and auxiliary arrays (e.g. ``time``, ``level``,
    ``y``, ``x``) are automatically skipped.

    Each source chunk is read exactly once; each destination chunk is written
    exactly once.  No intermediate temporary store is used.

    Parameters
    ----------
    src_path:
        Path to the source Zarr store (a group containing one or more arrays,
        as written by :func:`grib2zarr.initialise_zarr`).
    dst_path:
        Path for the output Zarr store.  Created (or overwritten) by this
        function.
    t_chunk:
        Chunk size along the leading time axis.  Defaults to ``24``.
    c_chunk:
        Chunk size along the second (vertical / channel) axis.  Defaults to
        ``None``, which means the full C dimension is used as one chunk.
        Reducing this value lowers the peak memory usage proportionally.
    spatial_chunk:
        Chunk size for both spatial axes (Y and X).  Defaults to ``100``.
    workers:
        Number of parallel worker processes used to rechunk variables.  Each
        variable is rechunked independently, so up to ``len(variables)``
        workers can be kept busy simultaneously.  Defaults to ``1``
        (sequential, no child processes spawned).
    zarr_format:
        Zarr format version of the destination store (``2`` or ``3``).
        Defaults to ``2``.  Sharding (see ``*_shard`` parameters) requires
        ``zarr_format=3``.
    t_shard, c_shard, spatial_shard:
        Optional shard sizes along the time, vertical and spatial axes
        respectively.  Each shard size must be a positive integer multiple
        of the corresponding chunk size.  When any shard size is provided,
        the destination arrays are written as Zarr v3 sharded arrays.
        Dimensions for which no shard size is specified default to the
        chunk size (i.e. no sharding along that dimension).  Requires
        ``zarr_format=3``.
    """
    _validate_shards(
        zarr_format, t_chunk, c_chunk, spatial_chunk,
        t_shard, c_shard, spatial_shard,
    )

    src_group = zarr.open_group(open_store(src_path), mode="r", zarr_format=2)
    dst_group = zarr.open_group(
        open_store(dst_path), mode="w", zarr_format=zarr_format
    )

    # Copy group-level attributes to the destination.
    dst_group.attrs.update(dict(src_group.attrs))

    array_names = [name for name, _ in src_group.arrays()]
    task_args = [
        (
            name, src_path, dst_path,
            t_chunk, c_chunk, spatial_chunk,
            zarr_format, t_shard, c_shard, spatial_shard,
        )
        for name in array_names
    ]

    if workers == 1:
        _log.info("rechunk  variables=%d", len(array_names))
        for args in task_args:
            done = _rechunk_variable_worker(args)
            _log.info("rechunk  finished '%s'", done)
    else:
        _log.info(
            "rechunk  workers=%d  variables=%d", workers, len(array_names)
        )
        with multiprocessing.Pool(processes=workers) as pool:
            for done in pool.map(_rechunk_variable_worker, task_args):
                _log.info("rechunk  finished '%s'", done)

    # Write consolidated metadata (.zmetadata) so tools like xarray can read
    # the store without scanning every array individually.
    zarr.consolidate_metadata(open_store(dst_path))


def _dimension_names(src: zarr.Array):
    """Return the dimension names for *src*, or ``None`` if unavailable.

    Zarr v3 stores dimension names in dedicated array metadata; Zarr v2 stores
    them in the ``_ARRAY_DIMENSIONS`` attribute (the xarray convention).  When
    copying from a v2 source to a v3 destination, xarray requires
    ``dimension_names`` to be set on the v3 array or ``xr.open_zarr`` fails
    with a ``KeyError`` about missing ``dimension_names`` metadata.
    """
    src_dim_names = getattr(src.metadata, "dimension_names", None)
    if src_dim_names:
        return tuple(src_dim_names)
    dims = src.attrs.get("_ARRAY_DIMENSIONS")
    if dims is not None:
        return tuple(dims)
    return None


def _copy_array(
    name: str,
    src: zarr.Array,
    dst_group: zarr.Group,
    chunks=None,
) -> None:
    """Copy an array to *dst_group*, optionally with new chunk sizes.

    Used for coordinate and auxiliary arrays.  When *chunks* is ``None`` the
    original chunk layout is preserved verbatim.  Pass an explicit tuple to
    rechunk the array as it is copied (e.g. 2-D spatial coordinate arrays).

    Parameters
    ----------
    name:
        Name of the array in the destination group.
    src:
        Source Zarr array.
    dst_group:
        Zarr group for the final output.
    chunks:
        Chunk shape for the destination array.  ``None`` keeps the source
        chunk layout unchanged.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        src_compressor = src.compressor

    # When copying into a Zarr v3 store, source (v2 numcodecs) compressors
    # cannot be passed through directly; fall back to zarr's default codecs.
    dst_format = getattr(dst_group.metadata, "zarr_format", 2)
    compressors = "auto" if dst_format == 3 else src_compressor

    create_kwargs = dict(
        shape=src.shape,
        chunks=src.chunks if chunks is None else chunks,
        dtype=src.dtype,
        compressors=compressors,
        fill_value=src.fill_value,
        overwrite=True,
    )
    if dst_format == 3:
        dim_names = _dimension_names(src)
        if dim_names is not None:
            create_kwargs["dimension_names"] = dim_names
    dst = dst_group.create_array(name, **create_kwargs)
    # Use Ellipsis indexing so that 0-D (scalar) arrays are handled correctly.
    dst[...] = src[...]
    dst.attrs.update(dict(src.attrs))


def _rechunk_array(
    name: str,
    src: zarr.Array,
    dst_group: zarr.Group,
    t_chunk: int,
    c_chunk: Optional[int],
    spatial_chunk: int,
    t_shard: Optional[int] = None,
    c_shard: Optional[int] = None,
    spatial_shard: Optional[int] = None,
) -> None:
    """Rechunk a single Zarr array using a single-pass algorithm.

    Parameters
    ----------
    name:
        Name of the array in the destination group.
    src:
        Source Zarr array.
    dst_group:
        Zarr group for the final output.
    t_chunk:
        Chunk size along the leading time axis.
    c_chunk:
        Chunk size along axis 1.  ``None`` means the full axis length.
    spatial_chunk:
        Chunk size for axes 2 and 3 (Y and X).
    t_shard, c_shard, spatial_shard:
        Optional shard sizes along the time, vertical and spatial axes.
        When any of these is provided the destination array is created as a
        Zarr v3 sharded array; dimensions without an explicit shard size
        default to the corresponding chunk size (i.e. one chunk per shard).

    Raises
    ------
    ValueError
        If *src* is not 4-dimensional.
    """
    if src.ndim != 4:
        raise ValueError(
            f"rechunk_zarr only supports 4-D arrays; "
            f"array '{name}' has {src.ndim} dimensions."
        )

    T, C, Y, X = src.shape
    effective_c_chunk = C if c_chunk is None else c_chunk

    # Determine whether to shard the destination array and, if so, the
    # per-dimension shard shape.  Dimensions without an explicit shard size
    # collapse to a single chunk per shard along that dimension.
    any_shard = any(s is not None for s in (t_shard, c_shard, spatial_shard))
    if any_shard:
        shards = (
            t_shard if t_shard is not None else t_chunk,
            c_shard if c_shard is not None else effective_c_chunk,
            spatial_shard if spatial_shard is not None else spatial_chunk,
            spatial_shard if spatial_shard is not None else spatial_chunk,
        )
    else:
        shards = None

    # Write in shard-aligned tiles when sharding is enabled so each shard is
    # written exactly once; otherwise fall back to chunk-aligned tiles.
    t_write = shards[0] if shards is not None else t_chunk
    c_write = shards[1] if shards is not None else effective_c_chunk
    spatial_write = shards[2] if shards is not None else spatial_chunk

    # Retrieve the source compressor in a zarr-version-agnostic way.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        src_compressor = src.compressor

    # When writing to a Zarr v3 destination, v2 numcodecs compressors cannot
    # be passed through directly; fall back to zarr's default codecs.
    dst_format = getattr(dst_group.metadata, "zarr_format", 2)
    compressors = "auto" if dst_format == 3 else src_compressor

    # ------------------------------------------------------------------
    # Create the destination store.
    # Chunks: (t_chunk, effective_c_chunk, spatial_chunk, spatial_chunk)
    # Shards: shards (v3 only, when sharding is enabled)
    # ------------------------------------------------------------------
    create_kwargs = dict(
        shape=src.shape,
        chunks=(t_chunk, effective_c_chunk, spatial_chunk, spatial_chunk),
        dtype=src.dtype,
        compressors=compressors,
        fill_value=src.fill_value,
        overwrite=True,
    )
    if shards is not None:
        create_kwargs["shards"] = shards
    if dst_format == 3:
        dim_names = _dimension_names(src)
        if dim_names is not None:
            create_kwargs["dimension_names"] = dim_names
    dst = dst_group.create_array(name, **create_kwargs)
    # Preserve the source array's attributes in the rechunked output.
    dst.attrs.update(dict(src.attrs))

    _log.info(
        "rechunk '%s'  shape=%s  src_chunks=%s  dst_chunks=%s  dst_shards=%s",
        name,
        src.shape,
        src.chunks,
        dst.chunks,
        shards,
    )

    # ------------------------------------------------------------------
    # Single pass: for each (time_block, level_block) read the entire
    # slab into a contiguous in-memory buffer of shape (tlen, clen, Y, X)
    # in one zarr call, then write to the destination in spatial tiles so
    # each destination chunk (or shard, when sharding is enabled) is
    # written exactly once.
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    for t0 in range(0, T, t_write):
        t1 = min(t0 + t_write, T)
        for c0 in range(0, C, c_write):
            c1 = min(c0 + c_write, C)
            buf = src[t0:t1, c0:c1, :, :]
            for y0 in range(0, Y, spatial_write):
                y1 = min(y0 + spatial_write, Y)
                for x0 in range(0, X, spatial_write):
                    x1 = min(x0 + spatial_write, X)
                    dst[t0:t1, c0:c1, y0:y1, x0:x1] = buf[:, :, y0:y1, x0:x1]

    _log.info(
        "rechunk '%s'  elapsed=%.1fs",
        name,
        time.perf_counter() - t_start,
    )


def _rechunk_variable_worker(task_args: tuple) -> str:
    """Rechunk (or copy) a single variable; suitable for use with :class:`multiprocessing.Pool`.

    This is a module-level function so that it can be pickled by
    :mod:`multiprocessing`.  It opens the source and destination Zarr stores
    by path (zarr objects are **not** passed across process boundaries).

    Parameters
    ----------
    task_args:
        A 10-tuple of
        ``(name, src_path, dst_path, t_chunk, c_chunk, spatial_chunk,
        zarr_format, t_shard, c_shard, spatial_shard)``.

    Returns
    -------
    str
        The variable *name* that was processed (useful for progress tracking).
    """
    (
        name,
        src_path,
        dst_path,
        t_chunk,
        c_chunk,
        spatial_chunk,
        zarr_format,
        t_shard,
        c_shard,
        spatial_shard,
    ) = task_args

    src_group = zarr.open_group(open_store(src_path), mode="r", zarr_format=2)
    src = src_group[name]
    dst_group = zarr.open_group(
        open_store(dst_path), mode="a", zarr_format=zarr_format
    )

    if src.ndim != 4:
        if src.ndim == 2:
            # Rechunk 2-D coordinate arrays (e.g. latitude, longitude) so that
            # their spatial tile size matches the spatial_chunk of the 4-D data
            # variables, keeping spatial access patterns aligned.
            dst_chunks = (
                min(spatial_chunk, src.shape[0]),
                min(spatial_chunk, src.shape[1]),
            )
            _log.info(
                "rechunk '%s'  shape=%s  src_chunks=%s  dst_chunks=%s",
                name,
                src.shape,
                src.chunks,
                dst_chunks,
            )
            _copy_array(name=name, src=src, dst_group=dst_group, chunks=dst_chunks)
        else:
            # Copy coordinate and auxiliary arrays (e.g. time, level, y, x)
            # verbatim – same chunks, same compressor, same attributes.
            _copy_array(name=name, src=src, dst_group=dst_group)
        return name

    _rechunk_array(
        name=name,
        src=src,
        dst_group=dst_group,
        t_chunk=t_chunk,
        c_chunk=c_chunk,
        spatial_chunk=spatial_chunk,
        t_shard=t_shard,
        c_shard=c_shard,
        spatial_shard=spatial_shard,
    )
    return name


def _parse_rechunk_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Rechunk a Zarr store written by grib2zarr into a layout optimised "
            "for time-series access using a single-pass algorithm that reads "
            "each source chunk exactly once with no intermediate temporary store."
        )
    )
    parser.add_argument(
        "src_path",
        metavar="SRC_PATH",
        help="Path to the source Zarr store to rechunk.",
    )
    parser.add_argument(
        "dst_path",
        metavar="DST_PATH",
        help="Path for the rechunked output Zarr store.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        metavar="N",
        dest="workers",
        help=(
            "Number of parallel worker processes to use for rechunking "
            "(default: 1).  Each variable is rechunked independently so "
            "performance scales with the number of variables up to this limit."
        ),
    )
    parser.add_argument(
        "--t-chunk",
        type=int,
        default=24,
        metavar="T",
        dest="t_chunk",
        help="Chunk size along the time axis (default: 24).",
    )
    parser.add_argument(
        "--c-chunk",
        type=int,
        default=None,
        metavar="C",
        dest="c_chunk",
        help=(
            "Chunk size along the vertical axis.  Defaults to the full C "
            "dimension (one chunk).  Reducing this value lowers peak memory "
            "usage proportionally."
        ),
    )
    parser.add_argument(
        "--spatial-chunk",
        type=int,
        default=100,
        metavar="S",
        dest="spatial_chunk",
        help="Chunk size for both spatial axes (Y and X) (default: 100).",
    )
    parser.add_argument(
        "--zarr-format",
        type=int,
        choices=(2, 3),
        default=2,
        dest="zarr_format",
        help=(
            "Zarr format version of the output store (default: 2).  Sharding "
            "options (--t-shard/--c-shard/--spatial-shard) require --zarr-format=3."
        ),
    )
    parser.add_argument(
        "--t-shard",
        type=int,
        default=None,
        metavar="T",
        dest="t_shard",
        help=(
            "Shard size along the time axis for Zarr v3 output.  Must be a "
            "positive integer multiple of --t-chunk.  Requires --zarr-format=3."
        ),
    )
    parser.add_argument(
        "--c-shard",
        type=int,
        default=None,
        metavar="C",
        dest="c_shard",
        help=(
            "Shard size along the vertical axis for Zarr v3 output.  Must be a "
            "positive integer multiple of --c-chunk.  Requires --zarr-format=3."
        ),
    )
    parser.add_argument(
        "--spatial-shard",
        type=int,
        default=None,
        metavar="S",
        dest="spatial_shard",
        help=(
            "Shard size for both spatial axes (Y and X) for Zarr v3 output.  "
            "Must be a positive integer multiple of --spatial-chunk.  Requires "
            "--zarr-format=3."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable INFO-level logging (shows per-variable timing, progress, etc.).",
    )
    return parser.parse_args(argv)


def cli() -> None:
    """Console-script entry point installed by ``pip install``."""
    args = _parse_rechunk_args(sys.argv[1:])
    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )
    rechunk_zarr(
        src_path=args.src_path,
        dst_path=args.dst_path,
        t_chunk=args.t_chunk,
        c_chunk=args.c_chunk,
        spatial_chunk=args.spatial_chunk,
        workers=args.workers,
        zarr_format=args.zarr_format,
        t_shard=args.t_shard,
        c_shard=args.c_shard,
        spatial_shard=args.spatial_shard,
    )


if __name__ == "__main__":
    cli()
