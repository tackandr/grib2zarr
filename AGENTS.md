# Agent instructions

Before changing behavior or moving code, read [ARCHITECTURE.md](ARCHITECTURE.md)
and the relevant module and tests.

## Repository conventions

- This is a flat Python project: modules live at the repository root and are
  listed in `pyproject.toml` under `tool.setuptools.py-modules`.
- Keep changes focused. Preserve the existing CLI entry points and library
  functions unless the task explicitly calls for changing them.
- Configuration is operator-supplied YAML. Preserve its existing structure and
  GRIB2/CF semantics when updating parsing or dataset construction.
- Output stores use Zarr format 2, including when accessed with the current
  Zarr library. Keep local and `s3://` paths working through `s3_store.py`.
- Parallel extraction relies on independent non-spatial chunks for concurrent
  writes. Do not weaken the chunk-alignment safety check without preserving
  write correctness.
- Match the existing Python style and add or update tests for behavior changes.

## Validation

- Run the test suite with `pytest`.
- No linter or formatter is configured in the repository; do not introduce one
  solely for a task.
- For packaging or CLI changes, also verify the relevant console-script or
  installation behavior when practical.
