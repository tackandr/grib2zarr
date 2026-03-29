"""
rechunk.py - Two-pass rechunking for Zarr stores.

Implements a manual two-pass rechunk that avoids the memory spike of a
single-pass copy when the source and target chunk layouts differ significantly
along multiple axes simultaneously.

Algorithm
---------
Pass 1 – *merge the leading axis, keep the vertical axis singleton, tile
the spatial axes*:

    temp chunks: (T_chunk, 1, spatial_chunk, spatial_chunk)

Each source chunk (one time-step, one level, full spatial extent) is read
once.  The write buffer for a single ``(t0:t1, c, :, :)`` block is of shape
``(T_chunk, 1, Y, X)`` which is small enough to hold in RAM.

Pass 2 – *merge the vertical axis into its final size*:

    final chunks: (T_chunk, C_chunk, spatial_chunk, spatial_chunk)

This pass reads the temp store, which is already tiled spatially, so each
read hits exactly one temp chunk.  The write buffer for a single tile is of
shape ``(T_chunk, C, spatial_chunk, spatial_chunk)`` which is also small.

Typical RAM usage for float32 data of shape ``(67, 66, 1069, 949)``:

* Pass 1 buffer: ~97 MiB  (24 × 1 × 1069 × 949 elements)
* Pass 2 buffer: ~60 MiB  (24 × 66 × 100 × 100 elements)

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
        tmp_path="tmp_rechunk.zarr",
        cleanup_tmp=True,
    )

Via the ``rechunk2zarr`` CLI::

    rechunk2zarr myfile.zarr rechunked.zarr --t-chunk 24 --spatial-chunk 100 --verbose
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
import tempfile
import time
import warnings
from typing import Optional

import numpy as np
import zarr

from s3_store import delete_store, open_store

_log = logging.getLogger(__name__)


def rechunk_zarr(
    src_path: str,
    dst_path: str,
    t_chunk: int = 24,
    c_chunk: Optional[int] = None,
    spatial_chunk: int = 100,
    tmp_path: Optional[str] = None,
    cleanup_tmp: bool = True,
    workers: int = 1,
) -> None:
    """Rechunk a 4-D Zarr array via a two-pass algorithm using a temporary store.

    The source array is expected to have shape ``(T, C, Y, X)`` – typically
    ``(n_timesteps, n_levels, n_lats, n_lons)``.  Only 4-D data arrays are
    rechunked; coordinate and auxiliary arrays (e.g. ``time``, ``level``,
    ``y``, ``x``) are automatically skipped.

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
    spatial_chunk:
        Chunk size for both spatial axes (Y and X).  Defaults to ``100``.
    tmp_path:
        Path for the intermediate temporary Zarr store.  When ``None`` a
        temporary directory inside ``dst_path``'s parent is created
        automatically and removed after use (controlled by *cleanup_tmp*).
    cleanup_tmp:
        When ``True`` (default) the temporary store is deleted after Pass 2
        completes.  Set to ``False`` to keep it for debugging.
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

    _own_tmp = tmp_path is None
    if _own_tmp:
        if dst_path.startswith("s3://"):
            # For S3 destinations keep the intermediate temp store on the local
            # scratch disk.  Writing pass-1 data to S3 one slice at a time would
            # be extremely slow; local I/O is fast and the data is discarded
            # after pass 2 completes.
            tmp_dir = tempfile.mkdtemp(prefix="rechunk_tmp_")
        else:
            tmp_dir = tempfile.mkdtemp(
                prefix="rechunk_tmp_", dir=os.path.dirname(os.path.abspath(dst_path))
            )
        tmp_path = tmp_dir
    # Initialise the shared temporary store so workers can open it in append mode.
    zarr.open_group(open_store(tmp_path), mode="w", zarr_format=2)

    array_names = [name for name, _ in src_group.arrays()]
    task_args = [
        (name, src_path, tmp_path, dst_path, t_chunk, c_chunk, spatial_chunk, cleanup_tmp)
        for name in array_names
    ]

    try:
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
    finally:
        if cleanup_tmp and _own_tmp:
            delete_store(tmp_path)

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
        overwrite=True,
    )
    # Use Ellipsis indexing so that 0-D (scalar) arrays are handled correctly.
    dst[...] = src[...]
    dst.attrs.update(dict(src.attrs))


def _rechunk_array(
    name: str,
    src: zarr.Array,
    tmp_group: zarr.Group,
    dst_group: zarr.Group,
    t_chunk: int,
    c_chunk: Optional[int],
    spatial_chunk: int,
) -> None:
    """Rechunk a single Zarr array using the two-pass algorithm.

    Parameters
    ----------
    name:
        Name of the array in both the temporary and destination groups.
    src:
        Source Zarr array.
    tmp_group:
        Zarr group used for the intermediate temporary store.
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
    # Create the temporary (pass-1) store
    # Chunks: (t_chunk, 1, spatial_chunk, spatial_chunk)
    # The vertical axis is kept as singleton so that each pass-1 write
    # only needs one level's data in the buffer at a time.
    # ------------------------------------------------------------------
    tmp = tmp_group.create_array(
        name,
        shape=src.shape,
        chunks=(t_chunk, 1, spatial_chunk, spatial_chunk),
        dtype=src.dtype,
        compressors=None,
        overwrite=True,
    )

    # ------------------------------------------------------------------
    # Create the destination (pass-2) store
    # Chunks: (t_chunk, c_chunk, spatial_chunk, spatial_chunk)
    # ------------------------------------------------------------------
    dst = dst_group.create_array(
        name,
        shape=src.shape,
        chunks=(t_chunk, effective_c_chunk, spatial_chunk, spatial_chunk),
        dtype=src.dtype,
        compressors=src_compressor,
        overwrite=True,
    )
    # Preserve the source array's attributes in the rechunked output.
    dst.attrs.update(dict(src.attrs))

    # ------------------------------------------------------------------
    # Pass 1: merge the time axis into t_chunk blocks; keep C = 1.
    # Each buffer is (min(t_chunk, T), 1, Y, X) – one level at a time.
    # ------------------------------------------------------------------
    _log.info(
        "rechunk '%s'  shape=%s  src_chunks=%s  dst_chunks=%s",
        name,
        src.shape,
        src.chunks,
        dst.chunks,
    )
    t_pass1_start = time.perf_counter()
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        tlen = t1 - t0
        for c in range(C):
            buf = np.empty((tlen, 1, Y, X), dtype=src.dtype)
            for k, t in enumerate(range(t0, t1)):
                buf[k, 0, :, :] = src[t, c, :, :]
            tmp[t0:t1, c : c + 1, :, :] = buf
    t_pass1 = time.perf_counter() - t_pass1_start

    # ------------------------------------------------------------------
    # Pass 2: merge the vertical axis; keep spatial tiles.
    # Each buffer is (min(t_chunk, T), C, spatial_chunk, spatial_chunk).
    # ------------------------------------------------------------------
    t_pass2_start = time.perf_counter()
    for t0 in range(0, T, t_chunk):
        t1 = min(t0 + t_chunk, T)
        for y0 in range(0, Y, spatial_chunk):
            y1 = min(y0 + spatial_chunk, Y)
            for x0 in range(0, X, spatial_chunk):
                x1 = min(x0 + spatial_chunk, X)
                buf = tmp[t0:t1, :, y0:y1, x0:x1]
                dst[t0:t1, :, y0:y1, x0:x1] = buf
    t_pass2 = time.perf_counter() - t_pass2_start

    _log.info(
        "rechunk '%s'  pass1=%.1fs  pass2=%.1fs  total=%.1fs",
        name,
        t_pass1,
        t_pass2,
        t_pass1 + t_pass2,
    )


def _rechunk_variable_worker(task_args: tuple) -> str:
    """Rechunk (or copy) a single variable; suitable for use with :class:`multiprocessing.Pool`.

    This is a module-level function so that it can be pickled by
    :mod:`multiprocessing`.  It opens the source, temporary and destination
    Zarr stores by path (zarr objects are **not** passed across process
    boundaries).

    Parameters
    ----------
    task_args:
        An 8-tuple of
        ``(name, src_path, tmp_path, dst_path, t_chunk, c_chunk,
        spatial_chunk, cleanup_tmp)``.

    Returns
    -------
    str
        The variable *name* that was processed (useful for progress tracking).
    """
    (
        name,
        src_path,
        tmp_path,
        dst_path,
        t_chunk,
        c_chunk,
        spatial_chunk,
        cleanup_tmp,
    ) = task_args

    src_group = zarr.open_group(open_store(src_path), mode="r", zarr_format=2)
    src = src_group[name]
    dst_group = zarr.open_group(open_store(dst_path), mode="a", zarr_format=2)

    if src.ndim != 4:
        # Copy coordinate and auxiliary arrays (e.g. time, level, y, x)
        # verbatim – same chunks, same compressor, same attributes.
        _copy_array(name=name, src=src, dst_group=dst_group)
        return name

    tmp_group = zarr.open_group(open_store(tmp_path), mode="a", zarr_format=2)
    _rechunk_array(
        name=name,
        src=src,
        tmp_group=tmp_group,
        dst_group=dst_group,
        t_chunk=t_chunk,
        c_chunk=c_chunk,
        spatial_chunk=spatial_chunk,
    )
    # Delete this variable's temp data as soon as Pass 2 is done so
    # that disk space is freed incrementally rather than only at the end.
    if cleanup_tmp and name in tmp_group:
        del tmp_group[name]

    return name


def _parse_rechunk_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Rechunk a Zarr store written by grib2zarr into a layout optimised "
            "for time-series access using a memory-efficient two-pass algorithm."
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
        "--tmp-path",
        default=None,
        metavar="TMP_PATH",
        dest="tmp_path",
        help=(
            "Path for the intermediate temporary Zarr store used between the two "
            "rechunk passes.  When not specified a temporary directory is created "
            "automatically inside the parent of DST_PATH (or on the local scratch "
            "disk for S3 destinations) and removed after rechunking completes."
        ),
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        default=False,
        dest="no_cleanup",
        help=(
            "Keep the intermediate temporary Zarr store instead of deleting it "
            "automatically after rechunking completes.  Useful for debugging."
        ),
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
            "dimension (one chunk)."
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
        tmp_path=args.tmp_path,
        cleanup_tmp=not args.no_cleanup,
        workers=args.workers,
    )


if __name__ == "__main__":
    cli()
