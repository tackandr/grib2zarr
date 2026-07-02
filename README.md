# grib2zarr

Extract parameters from a GRIB2 file and store them in a [Zarr](https://zarr.dev/) store using asyncio.

## Features

- Reads GRIB2 messages with [eccodes](https://github.com/ecmwf/eccodes-python)
- Extracts a user-specified list of parameters and stores each as a variable in a Zarr v2 store via [xarray](https://xarray.dev/)
- Uses an `asyncio` producer/consumer queue to decouple I/O-bound reading from writing
- Optional multi-threaded pipeline that reads several GRIB files and writes
  disjoint Zarr chunks in parallel (`--jobs`, `--writers`)

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
grib2zarr GRIB_FILE [GRIB_FILE ...] --config CONFIG [--output ZARR_PATH]
```

| Argument      | Description                                          | Default       |
|---------------|------------------------------------------------------|---------------|
| `GRIB_FILE`   | Path(s) to one or more input GRIB2 files (required)  | —             |
| `--config`    | Path to the YAML configuration file (required)       | —             |
| `--output`    | Destination Zarr store path                          | `myfile.zarr` |
| `--jobs`      | Number of parallel reader threads (one per input file) | `1`         |
| `--writers`   | Number of parallel writer threads                    | same as `--jobs` |

### Example

```bash
grib2zarr fc2026032709+001grib2_mbr000 fc2026032709+002grib2_mbr000 \
    --config config.yaml --output myfile.zarr
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

### Parallel pipeline

When `--jobs` or `--writers` is greater than `1`, a thread-based pipeline is
used instead: one reader thread per input GRIB file feeds a bounded
`queue.Queue`, and `--writers` writer threads pop items and write disjoint
`(time, level)` slabs to the Zarr store concurrently. Each writer opens the
Zarr store once at start-up rather than on every message. Because each
matched message maps to a unique `(variable, t_idx, z_idx, :, :)` region
computed from the message's own keys, output is deterministic regardless of
thread ordering.

To guarantee that concurrent writes never target the same Zarr chunk file,
the pipeline verifies that every non-spatial dimension of every variable has
`chunk: 1` in the config. If any variable has larger non-spatial chunks the
writer count is silently capped at `1` for correctness.

