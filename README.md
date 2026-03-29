# grib2zarr

Extract parameters from a GRIB2 file and store them in a [Zarr](https://zarr.dev/) store using asyncio.

## Features

- Reads GRIB2 messages with [eccodes](https://github.com/ecmwf/eccodes-python)
- Extracts a user-specified list of parameters and stores each as a variable in a Zarr v2 store via [xarray](https://xarray.dev/)
- Uses an `asyncio` producer/consumer queue to decouple I/O-bound reading from writing
- Optional **two-pass rechunking** (`--rechunk`) that restructures the output store into a layout optimised for time-series access without requiring the full dataset to fit in RAM
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

After installation a `grib2zarr` command is available on your `PATH`.

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

## Usage

```bash
grib2zarr GRIB_FILE [GRIB_FILE ...] --config CONFIG [--output ZARR_PATH] [--rechunk RECHUNK_PATH] [-v]
```

| Argument        | Description                                                                                      | Default       |
|-----------------|--------------------------------------------------------------------------------------------------|---------------|
| `GRIB_FILE`     | Path(s) to one or more input GRIB2 files (required)                                              | —             |
| `--config`      | Path to the YAML configuration file (required)                                                   | —             |
| `--output`      | Destination Zarr store path                                                                      | `myfile.zarr` |
| `--rechunk`     | When given, rechunk the output Zarr store into a new store at this path (see [Rechunking](#rechunking)) | —      |
| `-v`/`--verbose`| Enable INFO-level logging (rechunk timings, progress, etc.)                                      | off           |

### Example

```bash
grib2zarr fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \
    --config config.yaml --output myfile.zarr
```

### Example with rechunking and verbose output

```bash
grib2zarr fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \
    --config config.yaml \
    --output myfile.zarr \
    --rechunk myfile_rechunked.zarr \
    --verbose
```

The `--verbose` flag prints per-variable timing lines such as:

```
13:01:42 INFO rechunk 'temperature'  shape=(67, 66, 1069, 949)  src_chunks=(1, 1, 1069, 949)  dst_chunks=(24, 66, 100, 100)
13:02:44 INFO rechunk 'temperature'  pass1=42.3s  pass2=20.1s  total=62.4s
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
4. **Rechunk (optional)** – After writing, the store is rechunked into a new layout
   using a memory-efficient two-pass algorithm (see below).

## Rechunking

The initial Zarr store is written with source-aligned chunks that are optimal for
fast writing but not for downstream time-series reads.  The `--rechunk` option
runs a two-pass algorithm that converts the store into a layout with larger time
blocks and spatial tiles without loading the entire dataset into memory.

### Algorithm

**Pass 1** – merge the time axis into `t_chunk`-sized blocks while keeping the
vertical axis as singletons and the spatial axes tiled at `spatial_chunk`:

```
temp chunks: (t_chunk, 1, spatial_chunk, spatial_chunk)
```

**Pass 2** – merge the vertical axis into its final size, reading from the
temp store which is already spatially tiled:

```
final chunks: (t_chunk, c_chunk, spatial_chunk, spatial_chunk)
```

The temporary store is cleaned up variable-by-variable as each variable's
Pass 2 completes, so peak extra disk usage equals roughly one variable's
temp data at a time.

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

