"""
test_grib2zarr.py - Unit tests for the grib2zarr extraction pipeline.

Covers the performance-sensitive code paths changed to address the
slowdown observed when processing datasets with a large number of variables:

* ``_find_level_index`` – coordinate-value caching via ``_coord_cache``
* ``_find_time_index``  – step-value caching via ``_coord_cache``
* ``write_slice``       – accepts an already-open ``zarr.Group`` instead
                          of re-opening the store on every call
* ``consumer``          – opens the Zarr store once and reuses it
"""

import asyncio
import os
import tempfile

import numpy as np
import pytest
import xarray as xr
import zarr
import dask.array as da

import config_parser
from grib2zarr import (
    _find_level_index,
    _find_time_index,
    write_slice,
    consumer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmpdir, shape=(2, 3, 4, 5)):
    """Create a minimal Zarr store with one 4-D float32 variable."""
    T, C, Y, X = shape
    store_path = os.path.join(tmpdir, "test.zarr")
    data = da.zeros(shape, chunks=(1, 1, Y, X), dtype=np.float32)
    coord_time = xr.Variable(
        "time",
        np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[D]"),
        attrs={"axis": "T"},
    )
    coord_lev = xr.Variable("lev", np.arange(1, C + 1))
    ds = xr.Dataset(
        {"myvar": xr.DataArray(data, dims=["time", "lev", "y", "x"])},
        coords={"time": coord_time, "lev": coord_lev},
    )
    ds.to_zarr(store_path, mode="w", zarr_format=2, compute=False)
    return store_path, ds


# ---------------------------------------------------------------------------
# _find_level_index – caching
# ---------------------------------------------------------------------------

class TestFindLevelIndexCache:
    """_find_level_index should evaluate coord values only once per dim_ref."""

    def test_returns_correct_index(self):
        dim_ref = {"grib2": {"typeOfFirstFixedSurface": 105}, "values": [10, 20, 30]}
        dims = [dim_ref]
        msg_keys = {"typeOfFirstFixedSurface": 105}
        cache = {}
        assert _find_level_index(20, msg_keys, dims, cache) == 1

    def test_returns_none_for_missing_level(self):
        dim_ref = {"grib2": {"typeOfFirstFixedSurface": 105}, "values": [10, 20, 30]}
        dims = [dim_ref]
        msg_keys = {"typeOfFirstFixedSurface": 105}
        cache = {}
        assert _find_level_index(99, msg_keys, dims, cache) is None

    def test_coord_cache_populated_on_first_call(self):
        dim_ref = {"grib2": {"typeOfFirstFixedSurface": 105}, "values": [1, 2, 3]}
        dims = [dim_ref]
        msg_keys = {"typeOfFirstFixedSurface": 105}
        cache = {}
        _find_level_index(2, msg_keys, dims, cache)
        assert id(dim_ref) in cache
        assert cache[id(dim_ref)] == [1, 2, 3]

    def test_eval_values_called_once_across_repeated_calls(self, monkeypatch):
        """_eval_values must not be called again for the same dim_ref object."""
        dim_ref = {"grib2": {"typeOfFirstFixedSurface": 105}, "values": [1, 2, 3]}
        dims = [dim_ref]
        msg_keys = {"typeOfFirstFixedSurface": 105}
        cache = {}

        call_count = {"n": 0}
        original = config_parser._eval_values

        def counting_eval(values):
            call_count["n"] += 1
            return original(values)

        import grib2zarr as g2z
        monkeypatch.setattr(config_parser, "_eval_values", counting_eval)
        monkeypatch.setattr(g2z, "_eval_values", counting_eval)

        _find_level_index(1, msg_keys, dims, cache)
        n_after_first = call_count["n"]
        _find_level_index(2, msg_keys, dims, cache)
        n_after_second = call_count["n"]

        assert n_after_first == 1, "Expected exactly one _eval_values call on first lookup"
        assert n_after_second == n_after_first, (
            "_eval_values must not be called again for a cached dim_ref"
        )

    def test_no_match_when_grib2_keys_differ(self):
        dim_ref = {"grib2": {"typeOfFirstFixedSurface": 105}, "values": [1, 2, 3]}
        dims = [dim_ref]
        msg_keys = {"typeOfFirstFixedSurface": 1}  # different value
        cache = {}
        assert _find_level_index(1, msg_keys, dims, cache) is None


# ---------------------------------------------------------------------------
# _find_time_index – caching
# ---------------------------------------------------------------------------

class TestFindTimeIndexCache:
    """_find_time_index should evaluate step values only once per dim_ref."""

    def _make_time_dim(self):
        return {"reference_time": "2026-03-27T09:00:00", "values": [1, 2, 3]}

    def test_returns_correct_index(self):
        dim_ref = self._make_time_dim()
        dims = [dim_ref]
        # validity = reference + 2h → index 1
        msg_keys = {"validityDate": 20260327, "validityTime": 1100}
        cache = {}
        assert _find_time_index(msg_keys, dims, cache) == 1

    def test_returns_none_for_out_of_range_time(self):
        dim_ref = self._make_time_dim()
        dims = [dim_ref]
        # validity = reference + 100h → not in [1, 2, 3]
        msg_keys = {"validityDate": 20260331, "validityTime": 1300}
        cache = {}
        assert _find_time_index(msg_keys, dims, cache) is None

    def test_returns_zero_when_no_reference_time(self):
        dims = [{"grib2": {"typeOfFirstFixedSurface": 105}, "values": [1, 2, 3]}]
        cache = {}
        assert _find_time_index({}, dims, cache) == 0

    def test_eval_values_called_once_across_repeated_calls(self, monkeypatch):
        dim_ref = self._make_time_dim()
        dims = [dim_ref]
        msg_keys = {"validityDate": 20260327, "validityTime": 1100}
        cache = {}

        call_count = {"n": 0}
        original = config_parser._eval_values

        def counting_eval(values):
            call_count["n"] += 1
            return original(values)

        import grib2zarr as g2z
        monkeypatch.setattr(config_parser, "_eval_values", counting_eval)
        monkeypatch.setattr(g2z, "_eval_values", counting_eval)

        _find_time_index(msg_keys, dims, cache)
        n_after_first = call_count["n"]
        _find_time_index(msg_keys, dims, cache)
        n_after_second = call_count["n"]

        assert n_after_first == 1
        assert n_after_second == n_after_first, (
            "_eval_values must not be called again for a cached dim_ref"
        )


# ---------------------------------------------------------------------------
# write_slice – accepts open zarr.Group
# ---------------------------------------------------------------------------

class TestWriteSlice:
    """write_slice should write the correct data into the supplied open store."""

    def test_writes_correct_slice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path, ds = _make_store(tmpdir, shape=(2, 3, 4, 5))
            store = zarr.open(store_path, mode="r+")
            values = np.arange(20, dtype=np.float64)

            asyncio.run(write_slice(ds, "myvar", 0, 1, values, store))

            result = store["myvar"][0, 1, :, :]
            expected = values.reshape(4, 5).astype(np.float32)
            np.testing.assert_array_equal(result, expected)

    def test_does_not_overwrite_other_slices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path, ds = _make_store(tmpdir, shape=(2, 3, 4, 5))
            store = zarr.open(store_path, mode="r+")
            # Initialise with known sentinel values
            store["myvar"][:] = -1.0

            values = np.ones(20, dtype=np.float64) * 7.0
            asyncio.run(write_slice(ds, "myvar", 0, 0, values, store))

            # The written slice should be 7.0
            np.testing.assert_array_equal(store["myvar"][0, 0, :, :], 7.0)
            # All other time/level slices must remain -1.0
            np.testing.assert_array_equal(store["myvar"][1, 0, :, :], -1.0)
            np.testing.assert_array_equal(store["myvar"][0, 1, :, :], -1.0)


# ---------------------------------------------------------------------------
# consumer – opens the Zarr store only once
# ---------------------------------------------------------------------------

class TestConsumer:
    """consumer should open the Zarr store once regardless of message count."""

    def test_store_opened_once_for_multiple_messages(self, monkeypatch):
        """zarr.open must be called exactly once even when many slices are written."""
        open_call_count = {"n": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            store_path, ds = _make_store(tmpdir, shape=(2, 3, 4, 5))
            real_zarr_open = zarr.open

            def tracking_open(store_arg, mode=None, **kw):
                open_call_count["n"] += 1
                return real_zarr_open(store_arg, mode=mode, **kw)

            import grib2zarr as g2z
            monkeypatch.setattr(g2z.zarr, "open", tracking_open)

            async def _run():
                queue = asyncio.Queue()
                # Enqueue two messages for the same variable but different slices
                values = np.zeros(20, dtype=np.float64)
                queue.put_nowait(("myvar", 0, 0, values))
                queue.put_nowait(("myvar", 0, 1, values))
                queue.put_nowait(None)  # sentinel
                await consumer(queue, ds, store_path)

            asyncio.run(_run())

        assert open_call_count["n"] == 1, (
            f"zarr.open was called {open_call_count['n']} times; "
            "expected exactly 1 (store should be opened once and reused)"
        )

    def test_writes_all_messages_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store_path, ds = _make_store(tmpdir, shape=(2, 3, 4, 5))
            store = zarr.open(store_path, mode="r+")
            store["myvar"][:] = -1.0

            values_a = np.ones(20, dtype=np.float64) * 3.0
            values_b = np.ones(20, dtype=np.float64) * 9.0

            async def _run():
                queue = asyncio.Queue()
                queue.put_nowait(("myvar", 0, 0, values_a))
                queue.put_nowait(("myvar", 1, 2, values_b))
                queue.put_nowait(None)
                await consumer(queue, ds, store_path)

            asyncio.run(_run())

            store2 = zarr.open(store_path, mode="r")
            np.testing.assert_array_equal(store2["myvar"][0, 0, :, :], 3.0)
            np.testing.assert_array_equal(store2["myvar"][1, 2, :, :], 9.0)
            # Unwritten slice should still be -1.0
            np.testing.assert_array_equal(store2["myvar"][0, 1, :, :], -1.0)
