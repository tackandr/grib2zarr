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

Via the ``grib2zarr`` CLI::

    python grib2zarr.py fc*.grib2 --config config.yaml \\
        --output myfile.zarr --rechunk rechunked.zarr
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import time
import warnings
from typing import Optional

import numpy as np
import zarr

_log = logging.getLogger(__name__)


def rechunk_zarr(
    src_path: str,
    dst_path: str,
    t_chunk: int = 24,
    c_chunk: Optional[int] = None,
    spatial_chunk: int = 100,
    tmp_path: Optional[str] = None,
    cleanup_tmp: bool = True,
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
    """
    src_group = zarr.open_group(src_path, mode="r", zarr_format=2)
    dst_group = zarr.open_group(dst_path, mode="w", zarr_format=2)

    # Copy group-level attributes to the destination.
    dst_group.attrs.update(dict(src_group.attrs))

    _own_tmp = tmp_path is None
    if _own_tmp:
        tmp_dir = tempfile.mkdtemp(
            prefix="rechunk_tmp_", dir=os.path.dirname(os.path.abspath(dst_path))
        )
        tmp_path = tmp_dir
    tmp_group = zarr.open_group(tmp_path, mode="w", zarr_format=2)

    try:
        for name, src in src_group.arrays():
            if src.ndim != 4:
                # Copy coordinate and auxiliary arrays (e.g. time, level, y, x)
                # verbatim – same chunks, same compressor, same attributes.
                _copy_array(name=name, src=src, dst_group=dst_group)
                continue
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
    finally:
        if cleanup_tmp and _own_tmp:
            shutil.rmtree(tmp_path, ignore_errors=True)

    # Write consolidated metadata (.zmetadata) so tools like xarray can read
    # the store without scanning every array individually.
    zarr.consolidate_metadata(dst_path)


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
