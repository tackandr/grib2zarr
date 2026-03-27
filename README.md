# grib2zarr

Extract parameters from a GRIB2 file and store them in a [Zarr](https://zarr.dev/) store using asyncio.

## Features

- Reads GRIB2 messages with [eccodes](https://github.com/ecmwf/eccodes-python)
- Stores the u-wind parameter in a Zarr v2 store via [xarray](https://xarray.dev/)
- Uses an `asyncio` producer/consumer queue to decouple I/O-bound reading from writing

## Requirements

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** `eccodes` also requires the ECMWF eccodes C library to be installed on
> your system (`libeccodes-dev` on Debian/Ubuntu, `eccodes` via Homebrew on macOS).

## Usage

```bash
python grib2zarr.py <grib_file> [output_zarr_path]
```

| Argument          | Description                                      | Default        |
|-------------------|--------------------------------------------------|----------------|
| `grib_file`       | Path to the input GRIB2 file (required)          | —              |
| `output_zarr_path`| Destination Zarr store directory (optional)      | `myfile.zarr`  |

### Example

```bash
python grib2zarr.py fc2026032709+001grib2_mbr000 myfile.zarr
```

## How it works

1. **Initialise** – An empty Zarr store with shape `(66, 1069, 949)` and chunk
   size `(1, 1069, 949)` is created on disk before any data is read.
2. **Producer** – Reads GRIB2 messages one by one and puts them on an
   `asyncio.Queue`.  Only the raw grid values are decoded for `u` messages;
   all other messages are skipped cheaply.
3. **Consumer** – Picks messages off the queue and writes each `u` level slice
   directly into the Zarr store using `xr.Dataset.to_zarr(..., region=...)`.

## Grid assumptions

The defaults match a 66-level NWP model output on a 1069 × 949 grid.  If your
data has a different shape, edit `SHAPE` and `CHUNKS` at the top of
`grib2zarr.py`.
