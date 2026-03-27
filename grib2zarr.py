"""
grib2zarr.py - Extract GRIB2 messages and store them in Zarr format using asyncio.

This module reads a GRIB2 file, extracts u-wind parameter messages, and writes
the data into a Zarr store. A producer/consumer pattern is used with
asyncio.Queue to decouple reading from writing.

Usage:
    python grib2zarr.py <grib_file_path> [output_zarr_path]

Example:
    python grib2zarr.py fc2026032709+001grib2_mbr000 myfile.zarr
"""

import argparse
import asyncio
import sys

import dask.array as da
import xarray as xr
from eccodes import (
    codes_get,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
)

# Default output path
DEFAULT_ZARR_PATH = "myfile.zarr"

# Grid shape: (number of levels, ny, nx)
SHAPE = (66, 1069, 949)
CHUNKS = (1, 1069, 949)


def initialise_zarr(zarr_path: str) -> xr.Dataset:
    """Create an empty Zarr store pre-filled with the target dataset structure.

    Parameters
    ----------
    zarr_path:
        File-system path for the output Zarr store.

    Returns
    -------
    xr.Dataset
        The in-memory dataset whose Zarr store was just initialised.
    """
    data = da.empty(SHAPE, chunks=CHUNKS)
    u = xr.DataArray(data, dims=("z", "y", "x"))
    ds = xr.Dataset({"u": u})
    ds.to_zarr(zarr_path, mode="w", zarr_format=2, compute=False)
    return ds


async def read_grib(grib_file_path: str):
    """Async generator that yields one list per GRIB message.

    Each yielded item has the form::

        [shortName, level, dataDate, dataTime, values_or_None]

    ``values_or_None`` is the numpy array of grid values when
    ``shortName == "u"``, otherwise ``None`` so that we avoid reading
    unnecessary data.

    Parameters
    ----------
    grib_file_path:
        Path to the GRIB2 file to read.
    """
    with open(grib_file_path, "rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break  # No more messages

            short_name = codes_get(gid, "shortName")
            level = codes_get(gid, "level")
            data_date = codes_get(gid, "dataDate")
            data_time = codes_get(gid, "dataTime")
            values = codes_get_values(gid) if short_name == "u" else None

            codes_release(gid)

            yield [short_name, level, data_date, data_time, values]

            # Yield control back to the event loop between messages so that
            # the consumer can make progress while reading is ongoing.
            await asyncio.sleep(0)


async def write_u_slice(ds: xr.Dataset, message: list, zarr_path: str) -> None:
    """Write a single u-parameter slice into the Zarr store.

    Parameters
    ----------
    ds:
        The in-memory xarray Dataset that mirrors the Zarr store layout.
    message:
        A message list as produced by :func:`read_grib`.
    zarr_path:
        Path of the Zarr store to update.
    """
    _short_name, level, _data_date, _data_time, values = message
    z_idx = int(level)
    ny, nx = SHAPE[1], SHAPE[2]
    ds["u"][z_idx, :, :] = values.reshape(ny, nx)
    ds.isel(z=slice(z_idx, z_idx + 1)).to_zarr(
        zarr_path, mode="r+", region={"z": slice(z_idx, z_idx + 1)}
    )


async def producer(queue: asyncio.Queue, grib_file_path: str) -> None:
    """Read all GRIB messages and push them onto *queue*.

    A sentinel value of ``None`` is placed on the queue after all messages
    have been read to signal the consumer to stop.

    Parameters
    ----------
    queue:
        Shared asyncio queue.
    grib_file_path:
        Path to the GRIB2 file.
    """
    async for message in read_grib(grib_file_path):
        await queue.put(message)
    await queue.put(None)  # Sentinel – no more messages


async def consumer(
    queue: asyncio.Queue, ds: xr.Dataset, zarr_path: str
) -> None:
    """Consume messages from *queue* and write u-parameter data to Zarr.

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
        if message[0] == "u":
            await write_u_slice(ds, message, zarr_path)
        queue.task_done()


async def main(grib_file_path: str, zarr_path: str) -> None:
    """Orchestrate the producer/consumer pipeline.

    Parameters
    ----------
    grib_file_path:
        Path to the GRIB2 file to read.
    zarr_path:
        Destination Zarr store path.
    """
    ds = initialise_zarr(zarr_path)

    queue: asyncio.Queue = asyncio.Queue()
    producer_task = asyncio.create_task(producer(queue, grib_file_path))
    consumer_task = asyncio.create_task(consumer(queue, ds, zarr_path))

    await asyncio.gather(producer_task)
    await queue.join()
    await consumer_task

    print(f"Done. u-parameter written to '{zarr_path}'.")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract u-parameter from a GRIB2 file and store in Zarr."
    )
    parser.add_argument("grib_file", help="Path to the input GRIB2 file.")
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=DEFAULT_ZARR_PATH,
        help=f"Output Zarr store path (default: {DEFAULT_ZARR_PATH}).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    asyncio.run(main(args.grib_file, args.zarr_path))
