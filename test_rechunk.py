"""
test_rechunk.py - Unit tests for rechunk.py, covering Zarr v3 output and
per-dimension sharding options added to :func:`rechunk.rechunk_zarr`.
"""

import os
import tempfile

import numpy as np
import pytest
import xarray as xr
import zarr
import dask.array as da

from rechunk import _validate_shards, rechunk_zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_src(path, shape=(6, 4, 20, 20)):
    """Create a small Zarr v2 source store with one 4-D variable and coords."""
    T, C, Y, X = shape
    arr = np.arange(T * C * Y * X, dtype=np.float32).reshape(shape)
    data = da.from_array(arr, chunks=(1, 1, Y, X))
    ds = xr.Dataset(
        {"v": xr.DataArray(data, dims=["time", "lev", "y", "x"])},
        coords={"time": np.arange(T), "lev": np.arange(C)},
    )
    ds.to_zarr(path, mode="w", zarr_format=2)
    return arr


# ---------------------------------------------------------------------------
# _validate_shards
# ---------------------------------------------------------------------------

class TestValidateShards:
    def test_v2_no_shards_ok(self):
        _validate_shards(2, 3, 2, 10, None, None, None)

    def test_v3_no_shards_ok(self):
        _validate_shards(3, 3, 2, 10, None, None, None)

    def test_invalid_zarr_format(self):
        with pytest.raises(ValueError, match="zarr_format"):
            _validate_shards(4, 3, 2, 10, None, None, None)

    def test_shards_require_v3(self):
        with pytest.raises(ValueError, match="zarr_format=3"):
            _validate_shards(2, 3, 2, 10, 6, None, None)

    def test_shard_must_be_multiple_of_chunk(self):
        with pytest.raises(ValueError, match="multiple"):
            _validate_shards(3, 3, 2, 10, 7, None, None)
        with pytest.raises(ValueError, match="multiple"):
            _validate_shards(3, 3, 2, 10, None, None, 15)

    def test_shard_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            _validate_shards(3, 3, 2, 10, 0, None, None)

    def test_c_shard_requires_c_chunk(self):
        with pytest.raises(ValueError, match="c_chunk"):
            _validate_shards(3, 3, None, 10, None, 4, None)


# ---------------------------------------------------------------------------
# rechunk_zarr - end-to-end
# ---------------------------------------------------------------------------

class TestRechunkZarr:
    def test_default_output_is_zarr_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zarr")
            dst = os.path.join(tmp, "dst.zarr")
            expected = _make_src(src)
            rechunk_zarr(src, dst, t_chunk=3, c_chunk=2, spatial_chunk=10)

            g = zarr.open_group(dst, mode="r")
            assert g.metadata.zarr_format == 2
            assert g["v"].chunks == (3, 2, 10, 10)
            np.testing.assert_array_equal(g["v"][:], expected)

    def test_zarr_v3_without_sharding(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zarr")
            dst = os.path.join(tmp, "dst.zarr")
            expected = _make_src(src)
            rechunk_zarr(
                src, dst, t_chunk=3, c_chunk=2, spatial_chunk=10, zarr_format=3
            )

            g = zarr.open_group(dst, mode="r")
            assert g.metadata.zarr_format == 3
            assert g["v"].chunks == (3, 2, 10, 10)
            assert g["v"].shards is None
            np.testing.assert_array_equal(g["v"][:], expected)

    def test_zarr_v3_with_full_sharding(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zarr")
            dst = os.path.join(tmp, "dst.zarr")
            expected = _make_src(src)
            rechunk_zarr(
                src, dst,
                t_chunk=3, c_chunk=2, spatial_chunk=10,
                zarr_format=3,
                t_shard=6, c_shard=4, spatial_shard=20,
            )

            g = zarr.open_group(dst, mode="r")
            assert g.metadata.zarr_format == 3
            assert g["v"].chunks == (3, 2, 10, 10)
            assert g["v"].shards == (6, 4, 20, 20)
            np.testing.assert_array_equal(g["v"][:], expected)

    def test_zarr_v3_partial_sharding_defaults_missing_dims_to_chunk(self):
        """When only some *_shard args are given, other dims default to chunk size."""
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zarr")
            dst = os.path.join(tmp, "dst.zarr")
            expected = _make_src(src)
            rechunk_zarr(
                src, dst,
                t_chunk=3, c_chunk=2, spatial_chunk=10,
                zarr_format=3,
                spatial_shard=20,
            )

            g = zarr.open_group(dst, mode="r")
            assert g.metadata.zarr_format == 3
            # t and c shards default to their chunk sizes (no sharding along
            # those dims); spatial shard is explicitly 20.
            assert g["v"].shards == (3, 2, 20, 20)
            np.testing.assert_array_equal(g["v"][:], expected)

    def test_shard_requires_v3_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zarr")
            dst = os.path.join(tmp, "dst.zarr")
            _make_src(src)
            with pytest.raises(ValueError, match="zarr_format=3"):
                rechunk_zarr(src, dst, spatial_shard=20)

    def test_shard_not_multiple_of_chunk_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.zarr")
            dst = os.path.join(tmp, "dst.zarr")
            _make_src(src)
            with pytest.raises(ValueError, match="multiple"):
                rechunk_zarr(
                    src, dst,
                    t_chunk=3, c_chunk=2, spatial_chunk=10,
                    zarr_format=3, spatial_shard=15,
                )
