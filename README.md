# grib2zarr

Extract parameters from a GRIB2 file and store them in a [Zarr](https://zarr.dev/) store using asyncio.

## Features

- Reads GRIB2 messages with [eccodes](https://github.com/ecmwf/eccodes-python)
- Extracts a user-specified list of parameters and stores each as a variable in a Zarr v2 store via [xarray](https://xarray.dev/)
- Uses an `asyncio` producer/consumer queue to decouple I/O-bound reading from writing
- Separate **`rechunk2zarr`** CLI for optional single-pass rechunking that restructures a Zarr store into a layout optimised for time-series access
- Consolidated Zarr metadata (`.zmetadata`) written automatically so tools like xarray open the store without scanning every directory
- `--verbose` / `-v` flag for INFO-level progress and timing logs

## Requirements

> **Note:** `eccodes` requires the ECMWF eccodes C library to be installed on
> your system (`libeccodes-dev` on Debian/Ubuntu, `eccodes` via Homebrew on macOS).

## Installation

```bash
# Install the eccodes C library (Debian/Ubuntu)
sudo apt-get install libeccodes-dev

# Create a virtual environment and install the package in editable mode
python3 -m venv venv && source venv/bin/activate
pip install -e .
```

After installation a `grib2zarr` command and a `rechunk2zarr` command are available on your `PATH`.

## Container (Podman / Docker)

Build the image:

```bash
podman build -t grib2zarr -f Containerfile .
```

Run the container, mounting the directory that contains your GRIB2 files and
config:

```bash
podman run --rm \
    -v /path/to/data:/data \
    grib2zarr \
    /data/fc2026032709+001grib2_mbr000 \
    --config /data/config.yaml \
    --output /data/myfile.zarr
```

To rechunk inside the container, override the entrypoint:

```bash
podman run --rm \
    -v /path/to/data:/data \
    --entrypoint rechunk2zarr \
    grib2zarr \
    /data/myfile.zarr /data/myfile_rechunked.zarr
```

## Usage

### grib2zarr

```bash
grib2zarr GRIB_FILE [GRIB_FILE ...] --config CONFIG [--output ZARR_PATH] [-v]
```

| Argument        | Description                                                                                      | Default       |
|-----------------|--------------------------------------------------------------------------------------------------|---------------|
| `GRIB_FILE`     | Path(s) to one or more input GRIB2 files (required)                                              | —             |
| `--config`      | Path to the YAML configuration file (required)                                                   | —             |
| `--output`      | Destination Zarr store path                                                                      | `myfile.zarr` |
| `-v`/`--verbose`| Enable INFO-level logging (progress, timing, etc.)                                               | off           |

### Example

```bash
grib2zarr fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \
    --config config.yaml --output myfile.zarr
```

### rechunk2zarr

After extracting to a Zarr store with `grib2zarr`, use `rechunk2zarr` to rechunk it into a layout optimised for time-series access:

```bash
rechunk2zarr SRC_PATH DST_PATH [-j N]
             [--t-chunk T] [--c-chunk C] [--spatial-chunk S] [-v]
```

| Argument           | Description                                                                                     | Default      |
|--------------------|-------------------------------------------------------------------------------------------------|--------------|
| `SRC_PATH`         | Path to the source Zarr store (required)                                                        | —            |
| `DST_PATH`         | Path for the rechunked output Zarr store (required)                                             | —            |
| `-j`/`--jobs`      | Number of parallel worker processes (each variable is independent)                              | `1`          |
| `--t-chunk`        | Chunk size along the time axis                                                                  | `24`         |
| `--c-chunk`        | Chunk size along the vertical axis (defaults to full C dimension; smaller values reduce RAM)    | full C       |
| `--spatial-chunk`  | Chunk size for both spatial axes (Y and X)                                                      | `100`        |
| `-v`/`--verbose`   | Enable INFO-level logging (per-variable timing, progress, etc.)                                 | off          |

### Example with rechunking

```bash
# Step 1 – extract GRIB2 files to Zarr
grib2zarr fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \
    --config config.yaml --output myfile.zarr

# Step 2 – rechunk the Zarr store
rechunk2zarr myfile.zarr myfile_rechunked.zarr --verbose
```

## How it works

1. **Initialise** – An empty Zarr store is created from the YAML configuration: shape,
   chunk layout, coordinates, CRS and CF attributes are all derived from the config.
2. **Producer** – Reads GRIB2 messages one by one and puts matched messages on an
   `asyncio.Queue`.  Variables are identified by their GRIB2
   `discipline`/`parameterCategory`/`parameterNumber` keys; unmatched messages are
   skipped cheaply.
3. **Consumer** – Picks messages off the queue and writes each level slice directly
   into the Zarr store.
4. **Rechunk (optional)** – After writing, run `rechunk2zarr` to rechunk the store
   into a new layout using a single-pass algorithm (see below).

## Rechunking

The initial Zarr store is written with source-aligned chunks that are optimal for
fast writing but not for downstream time-series reads.  The `rechunk2zarr` CLI
uses a single-pass algorithm that converts the store into a layout with larger time
blocks and spatial tiles.

### Algorithm

For each ``(time_block, level_block)`` combination:

1. Read all source chunks for that block into a contiguous in-memory buffer of shape
   ``(tlen, clen, Y, X)`` where ``tlen ≤ t_chunk`` and ``clen ≤ c_chunk``.
2. Write the buffer to the destination in aligned spatial tiles so that each
   destination chunk is written exactly once.

Each source chunk is read exactly once.  No intermediate temporary store is written
to disk.  Peak memory usage is proportional to ``t_chunk × c_chunk × Y × X``; use
``--c-chunk`` to reduce it when memory is constrained.

### Defaults

| Parameter      | Default | Description                          |
|----------------|---------|--------------------------------------|
| `t_chunk`      | 24      | Chunk size along the time axis       |
| `c_chunk`      | full C  | Chunk size along the vertical axis   |
| `spatial_chunk`| 100     | Chunk size for both spatial axes     |

### Using rechunk as a library

```python
from rechunk import rechunk_zarr

rechunk_zarr(
    src_path="myfile.zarr",
    dst_path="myfile_rechunked.zarr",
    t_chunk=24,
    c_chunk=None,       # defaults to full C axis
    spatial_chunk=100,
)
```

