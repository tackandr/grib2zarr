"""
grib2zarr.py - Extract GRIB2 messages and store them in Zarr format using asyncio.

This module reads a GRIB2 file, extracts parameters described by a YAML
configuration file, and writes the data into a Zarr store.  A
producer/consumer pattern is used with asyncio.Queue to decouple reading
from writing.

Variables are identified by their GRIB2 discipline / parameterCategory /
parameterNumber keys.  Dataset shape, chunk layout, coordinates, CRS and
CF attributes are all derived from the configuration file.

Usage:
    python grib2zarr.py <grib_file_path> [output_zarr_path] --config CONFIG

Example:
    python grib2zarr.py fc2026032709+001grib2_mbr000 myfile.zarr --config config.yaml
"""

import argparse
import asyncio
import sys
from typing import List

import xarray as xr
from eccodes import (
    codes_get,
    codes_get_values,
    codes_grib_new_from_file,
    codes_release,
)

from config_parser import build_dataset, load_config, _eval_values

# Default output path
DEFAULT_ZARR_PATH = "myfile.zarr"


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
    ds.to_zarr(zarr_path, mode="w", zarr_format=2, compute=False)
    return ds


def _build_var_matcher(config: dict) -> list:
    """Build a list of variable matchers from a parsed config.

    Each entry is a tuple ``(var_name, grib2_keys, dims)`` where *grib2_keys*
    is a dict of the GRIB2 identification keys for that variable and *dims* is
    the list of dimension reference dicts as written in the config.

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
        matchers.append((var_name, grib2_keys, dims))
    return matchers


def _find_level_index(level: int, type_of_first_fixed_surface: int, dims: list) -> int:
    """Return the index of *level* within the matching vertical coordinate.

    Searches *dims* for a dimension whose ``grib2.typeOfFirstFixedSurface``
    matches *type_of_first_fixed_surface*, then looks up *level* in that
    dimension's value list.

    If the level value is found in the list the list-position index is
    returned.  Otherwise, for 1-based level numbering (typical for hybrid
    levels), ``level - 1`` is used as a fallback.  Returns 0 when no
    matching dimension is found.

    Parameters
    ----------
    level:
        GRIB ``level`` key value for the current message.
    type_of_first_fixed_surface:
        GRIB ``typeOfFirstFixedSurface`` key value for the current message.
    dims:
        List of dimension reference dicts from the variable configuration.

    Returns
    -------
    int
        Zero-based level index into the vertical coordinate array.
    """
    for dim_ref in dims:
        if not isinstance(dim_ref, dict):
            continue
        grib2 = dim_ref.get("grib2", {})
        if grib2.get("typeOfFirstFixedSurface") == type_of_first_fixed_surface:
            values = _eval_values(dim_ref.get("values", []))
            if level in values:
                # Exact match: return the list position
                return values.index(level)
            # Fallback for 1-based level numbering (e.g. hybrid levels
            # numbered 1..N): only applied when `level` was not found above
            if 1 <= level <= len(values):
                return level - 1
    return 0


async def read_grib(grib_file_path: str, matchers: list):
    """Async generator yielding matched GRIB messages.

    Each yielded item is a tuple::

        (var_name, t_idx, z_idx, values)

    where *t_idx* is always 0 (single time-step assumption), *z_idx* is the
    zero-based index into the variable's vertical coordinate, and *values* is
    the flat numpy array of grid values.

    Only messages that match at least one variable in *matchers* are yielded.

    Parameters
    ----------
    grib_file_path:
        Path to the GRIB2 file to read.
    matchers:
        List produced by :func:`_build_var_matcher`.
    """
    with open(grib_file_path, "rb") as f:
        while True:
            gid = codes_grib_new_from_file(f)
            if gid is None:
                break

            discipline = codes_get(gid, "discipline")
            param_category = codes_get(gid, "parameterCategory")
            param_number = codes_get(gid, "parameterNumber")
            tofss = codes_get(gid, "typeOfFirstFixedSurface")
            level = codes_get(gid, "level")

            matched = None
            for var_name, grib2_keys, dims in matchers:
                if (
                    discipline == grib2_keys.get("discipline")
                    and param_category == grib2_keys.get("parameterCategory")
                    and param_number == grib2_keys.get("parameterNumber")
                ):
                    matched = (var_name, dims)
                    break

            if matched is not None:
                var_name, dims = matched
                values = codes_get_values(gid)
                z_idx = _find_level_index(level, tofss, dims)
                codes_release(gid)
                yield (var_name, 0, z_idx, values)
            else:
                codes_release(gid)

            # Yield control back to the event loop between messages so that
            # the consumer can make progress while reading is ongoing.
            await asyncio.sleep(0)


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
    full; the ``steps`` and vertical dimensions are written at the indices
    given by *t_idx* and *z_idx* respectively.

    Parameters
    ----------
    ds:
        In-memory dataset produced by :func:`initialise_zarr`.
    var_name:
        Name of the target data variable in *ds*.
    t_idx:
        Index along the ``steps`` (time) dimension.
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

    # Reshape flat values to (ny, nx)
    ny = da_var.sizes[dim_names[-2]]
    nx = da_var.sizes[dim_names[-1]]
    grid = values.reshape(ny, nx)

    # Build the numpy index tuple and the zarr region dict
    np_idx = []
    region = {}
    for dim in dim_names:
        if dim == "steps":
            np_idx.append(t_idx)
            region[dim] = slice(t_idx, t_idx + 1)
        elif dim in spatial_dims:
            np_idx.append(slice(None))
            region[dim] = slice(None)
        else:
            # Vertical dimension
            np_idx.append(z_idx)
            region[dim] = slice(z_idx, z_idx + 1)

    # Assign to in-memory array
    ds[var_name].values[tuple(np_idx)] = grid

    # Write the slice to the Zarr store.
    # Each GRIB message encodes a single level, so there is at most one
    # vertical dimension per variable; the dict comprehension is safe here.
    isel_dict = {
        d: slice(t_idx, t_idx + 1) if d == "steps" else slice(z_idx, z_idx + 1)
        for d in dim_names
        if d not in spatial_dims
    }
    ds[[var_name]].isel(isel_dict).to_zarr(zarr_path, mode="r+", region=region)


async def producer(
    queue: asyncio.Queue,
    grib_file_path: str,
    matchers: list,
) -> None:
    """Read GRIB messages and push them onto *queue*.

    A sentinel value of ``None`` is placed on the queue after all messages
    have been read to signal the consumer to stop.

    Parameters
    ----------
    queue:
        Shared asyncio queue.
    grib_file_path:
        Path to the GRIB2 file.
    matchers:
        List produced by :func:`_build_var_matcher`.
    """
    async for message in read_grib(grib_file_path, matchers):
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


async def main(grib_file_path: str, zarr_path: str, config_path: str) -> None:
    """Orchestrate the producer/consumer pipeline.

    Parameters
    ----------
    grib_file_path:
        Path to the GRIB2 file to read.
    zarr_path:
        Destination Zarr store path.
    config_path:
        Path to the YAML configuration file describing the target dataset.
    """
    config = load_config(config_path)
    ds = initialise_zarr(zarr_path, config)
    matchers = _build_var_matcher(config)
    var_names = [m[0] for m in matchers]

    queue: asyncio.Queue = asyncio.Queue()
    producer_task = asyncio.create_task(producer(queue, grib_file_path, matchers))
    consumer_task = asyncio.create_task(consumer(queue, ds, zarr_path))

    await asyncio.gather(producer_task)
    await queue.join()
    await consumer_task

    print(
        f"Done. Variables {', '.join(var_names)} written to '{zarr_path}' "
        f"using config '{config_path}'."
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Extract parameters from a GRIB2 file and store them in Zarr."
        )
    )
    parser.add_argument("grib_file", help="Path to the input GRIB2 file.")
    parser.add_argument(
        "zarr_path",
        nargs="?",
        default=DEFAULT_ZARR_PATH,
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args(sys.argv[1:])
    asyncio.run(main(args.grib_file, args.zarr_path, args.config))

