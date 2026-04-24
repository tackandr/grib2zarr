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

Via the ``rechunk2zarr`` CLI::

    rechunk2zarr myfile.zarr rechunked.zarr --t-chunk 24 --spatial-chunk 100 --verbose
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


def rechunk_zarr(
    src_path: str,
    dst_path: str,
    t_chunk: int = 24,
    c_chunk: Optional[int] = None,
    spatial_chunk: int = 100,
    workers: int = 1,
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
    """
    src_group = zarr.open_group(open_store(src_path), mode="r", zarr_format=2)
    dst_group = zarr.open_group(open_store(dst_path), mode="w", zarr_format=2)

    # Copy group-level attributes to the destination.
    dst_group.attrs.update(dict(src_group.attrs))

    array_names = [name for name, _ in src_group.arrays()]
    task_args = [
        (name, src_path, dst_path, t_chunk, c_chunk, spatial_chunk)
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


def _copy_array(name: str, src: zarr.Array, dst_group: zarr.Group) -> None:
    """Copy an array verbatim (data, chunks, compressor, attributes) to *dst_group*.

    Used for coordinate and auxiliary arrays that do not need rechunking.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        src_compressor = src.compressor

    dst = dst_group.create_array(
        name,
        shape=src.shape,
        chunks=src.chunks,
        dtype=src.dtype,
        compressors=src_compressor,
        fill_value=src.fill_value,
        overwrite=True,
    )
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

    # Retrieve the source compressor in a zarr-version-agnostic way.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        src_compressor = src.compressor

    # ------------------------------------------------------------------
    # Create the destination store.
    # Chunks: (t_chunk, effective_c_chunk, spatial_chunk, spatial_chunk)
    # ------------------------------------------------------------------
    dst = dst_group.create_array(
        name,
        shape=src.shape,
        chunks=(t_chunk, effective_c_chunk, spatial_chunk, spatial_chunk),
        dtype=src.dtype,
        compressors=src_compressor,
        fill_value=src.fill_value,
        overwrite=True,
    )
    # Preserve the source array's attributes in the rechunked output.
    dst.attrs.update(dict(src.attrs))

    _log.info(
        "rechunk '%s'  shape=%s  src_chunks=%s  dst_chunks=%s",
        name,
        src.shape,
        src.chunks,
        dst.chunks,
    )

    # ------------------------------------------------------------------
    # Single pass: for each (time_block, level_block) read the entire
    # slab into a contiguous in-memory buffer of shape (tlen, clen, Y, X)
    # in one zarr call, then write to the destination in spatial tiles so
    # each destination chunk is written exactly once.
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        for c0 in range(0, C, effective_c_chunk):
            c1 = min(c0 + effective_c_chunk, C)
            buf = src[t0:t1, c0:c1, :, :]
            for y0 in range(0, Y, spatial_chunk):
                y1 = min(y0 + spatial_chunk, Y)
                for x0 in range(0, X, spatial_chunk):
                    x1 = min(x0 + spatial_chunk, X)
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
        A 6-tuple of
        ``(name, src_path, dst_path, t_chunk, c_chunk, spatial_chunk)``.

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
    ) = task_args

    src_group = zarr.open_group(open_store(src_path), mode="r", zarr_format=2)
    src = src_group[name]
    dst_group = zarr.open_group(open_store(dst_path), mode="a", zarr_format=2)

    if src.ndim != 4:
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
    )


if __name__ == "__main__":
    cli()
