# grib2zarr

Extract parameters from a GRIB2 file and store them in a [Zarr](https://zarr.dev/) store using asyncio.

## Features

- Reads GRIB2 messages with [eccodes](https://github.com/ecmwf/eccodes-python)
- Extracts a user-specified list of parameters and stores each as a variable in a Zarr v2 store via [xarray](https://xarray.dev/)
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
python grib2zarr.py <grib_file> [output_zarr_path] [--params PARAM ...]
```

| Argument           | Description                                                  | Default        |
|--------------------|--------------------------------------------------------------|----------------|
| `grib_file`        | Path to the input GRIB2 file (required)                      | —              |
| `output_zarr_path` | Destination Zarr store directory (optional)                  | `myfile.zarr`  |
| `--params`         | One or more GRIB shortNames to extract (optional, repeatable)| `u`            |

### Example

```bash
# Extract only u (default)
python grib2zarr.py fc2026032709+001grib2_mbr000 myfile.zarr

# Extract u and v
python grib2zarr.py fc2026032709+001grib2_mbr000 myfile.zarr --params u v

# Extract u, v, and temperature
python grib2zarr.py fc2026032709+001grib2_mbr000 myfile.zarr --params u v t
```

## How it works

1. **Initialise** – An empty Zarr store is created with one variable per requested
   parameter, each with shape `(66, 1069, 949)` and chunk size `(1, 1069, 949)`.
2. **Producer** – Reads GRIB2 messages one by one and puts them on an
   `asyncio.Queue`.  Grid values are only decoded for messages whose shortName
   matches one of the requested parameters; all other messages are skipped cheaply.
3. **Consumer** – Picks messages off the queue and calls `write_slice()` for every
   matching parameter, writing each level slice directly into the Zarr store using
   `xr.DataArray.to_zarr(..., region=...)`.

## Grid assumptions

The defaults match a 66-level NWP model output on a 1069 × 949 grid.  If your
data has a different shape, edit `SHAPE` and `CHUNKS` at the top of
`grib2zarr.py`.
