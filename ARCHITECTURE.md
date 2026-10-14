# Architecture

`grib2zarr` converts configured fields in GRIB2 files into a Zarr v2 dataset.
The project is a flat Python application: the modules are installed from the
repository root and the package exposes the `grib2zarr` and `rechunk2zarr`
console commands.

## Extraction flow

1. `grib2zarr.cli` parses input paths, the YAML config path, output path, and
   reader/writer counts. `main` loads the config and creates the output dataset.
2. `config_parser.load_config` reads YAML. `build_dataset` turns its
   coordinates, geometries, and variables into an xarray Dataset backed by
   lazy Dask arrays. Coordinate expressions in string form are evaluated in a
   restricted namespace; configuration files are expected to be
   operator-supplied.
3. `grib2zarr._build_var_matcher` combines each variable's GRIB2 keys with
   identifying keys from its dimensions. `_read_grib_file` scans messages,
   matches variables, maps validity times and levels to configured indices,
   and yields matched grid values.
4. With the default single reader and writer, an asyncio producer sends matched
   messages through a queue to a consumer. With parallelism enabled, reader
   threads feed a bounded queue and writer threads write to the store.
5. `write_slice` writes each matched grid to its variable's time/level position.
   `initialise_zarr` creates the store with `zarr_format=2`, coordinates,
   metadata, and configured chunking.
6. `s3_store.open_store` passes local paths through and creates an S3-backed
   Zarr store for `s3://` paths. S3 credentials and an optional custom endpoint
   are obtained from environment variables; see `s3_store.py`.

## Rechunking flow

`rechunk2zarr` is implemented in `rechunk.py`. It copies group metadata and
non-4-D coordinate/auxiliary arrays, and rechunks 4-D data arrays expected in
`(T, C, Y, X)` order. The single-pass algorithm reads time/level blocks into
memory and writes spatial tiles to the destination. Optional worker processes
handle separate arrays. `--c-chunk` controls the vertical block size and thus
the peak buffer size. Both source and destination paths use `s3_store`.

## Module responsibilities

- `grib2zarr.py`: GRIB2 scanning and matching, time/level indexing, store
  initialization, serial and threaded extraction pipelines, and CLI.
- `config_parser.py`: YAML loading, coordinate evaluation, CF/CRS coordinate
  construction, and xarray Dataset creation.
- `s3_store.py`: local/S3 Zarr store selection and store deletion.
- `rechunk.py`: Zarr copy/rechunk implementation and `rechunk2zarr` CLI.
- `config.yaml`: example dataset configuration, including CF metadata, GRIB2
  match keys, dimensions, and chunk settings.
- `test_grib2zarr.py`: tests for extraction matching/indexing, slice writes,
  and the serial consumer's store reuse.

## Correctness constraints

- GRIB2 fields are matched by configured identification keys; vertical
  dimension keys and valid levels distinguish fields that otherwise share
  parameter keys.
- A message's validity date/time and level must map to configured coordinate
  indices before it is written.
- Each extracted message writes a complete spatial grid at a single
  variable/time/level position. Threaded writes are safe only when the
  non-spatial chunks are size one; `_check_chunk_alignment` enforces this by
  limiting writers when needed.
- Rechunking is for four-dimensional data arrays in `(T, C, Y, X)` order.
  Coordinate arrays are copied, with two-dimensional spatial coordinates
  aligned to the requested spatial chunk size.
