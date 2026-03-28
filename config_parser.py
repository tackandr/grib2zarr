"""
config_parser.py - Parse a YAML dataset configuration and build an empty xarray Dataset.

The YAML configuration describes a CF-conventions dataset structure including
coordinates (with optional auxiliary parameter variables), geometries (CRS /
grid_mapping), and data variables with their GRIB2 identification keys and
dimension references.

Usage example::

    from config_parser import load_config, build_dataset

    config = load_config("config.yaml")
    ds = build_dataset(config)
    ds.to_zarr("output.zarr", mode="w", compute=False)
"""

from __future__ import annotations

import yaml
import numpy as np
import xarray as xr
import dask.array as da


def load_config(yaml_path: str) -> dict:
    """Load and parse a YAML configuration file.

    Parameters
    ----------
    yaml_path:
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _eval_values(values) -> list:
    """Return a plain Python list from *values*.

    *values* may be:

    * A Python list (returned as-is).
    * A string containing a valid Python list expression, e.g.
      ``'[i * 2500 for i in range(949)]'``, which is evaluated with
      :func:`eval`.  Only list-literal and list-comprehension expressions
      are expected; arbitrary code execution is a user responsibility
      because the YAML config is a trusted, operator-supplied file.

    Parameters
    ----------
    values:
        Raw value from the YAML configuration.

    Returns
    -------
    list
        Evaluated list of coordinate values.
    """
    if isinstance(values, str):
        # Restrict evaluation to a safe namespace that only exposes `range`
        # and arithmetic builtins so that arbitrary code from the YAML file
        # cannot be executed.
        safe_globals: dict = {"__builtins__": {}, "range": range}
        result = eval(values, safe_globals)  # nosec – restricted namespace
        return list(result)
    return list(values)


def build_dataset(config: dict) -> xr.Dataset:
    """Build an empty xarray Dataset from a parsed YAML configuration.

    The returned dataset contains:

    * **Coordinates** – one per entry in ``config['coordinates']``, with CF
      attributes.  Auxiliary parameter variables (e.g. *a* and *b*
      coefficients for a hybrid sigma-pressure coordinate) are added as
      non-index coordinates.
    * **Grid-mapping variables** – one scalar integer variable per entry in
      ``config['geometries']``, carrying CRS metadata as attributes following
      the CF grid_mapping convention.
    * **Data variables** – one per entry in ``config['variables']``, backed
      by a lazy :mod:`dask` array of the correct shape and chunk layout
      derived from the dimension references in the configuration.

    Parameters
    ----------
    config:
        Dictionary produced by :func:`load_config`.

    Returns
    -------
    xr.Dataset
        Empty (lazily allocated) dataset ready to be written to a Zarr store.
    """
    coords: dict[str, xr.Variable] = {}
    data_vars: dict[str, xr.Variable | xr.DataArray] = {}

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    for coord_entry in config.get("coordinates", []):
        for _, info in coord_entry.items():
            name = info["name"]
            values = np.array(_eval_values(info["values"]))
            cf_attrs = dict(info.get("cf", {}))
            coords[name] = xr.Variable(name, values, attrs=cf_attrs)

            # Auxiliary parameter variables (e.g. a/b for hybrid levels)
            for param in info.get("parameters", []):
                param_name = param["name"]
                param_values = np.array(_eval_values(param["values"]))
                param_cf = dict(param.get("cf", {}))
                param_dims = [
                    d["name"] if isinstance(d, dict) else name
                    for d in param.get("dims", [{"name": name}])
                ]
                coords[param_name] = xr.Variable(
                    param_dims, param_values, attrs=param_cf
                )

    # ------------------------------------------------------------------
    # Geometries (grid_mapping / CRS)
    # ------------------------------------------------------------------
    for geom_entry in config.get("geometries", []):
        for _, info in geom_entry.items():
            geom_name = info["name"]
            crs_attrs = dict(info.get("cf", {}).get("crs", {}))
            # Scalar integer variable – standard CF grid_mapping convention
            data_vars[geom_name] = xr.Variable((), np.int32(0), attrs=crs_attrs)

    # ------------------------------------------------------------------
    # Data variables
    # ------------------------------------------------------------------
    for var in config.get("variables", []):
        var_name = var["name"]
        cf_attrs = dict(var.get("cf", {}))

        # Attach grid_mapping attribute when a geometry is referenced
        geom = var.get("geometry")
        if isinstance(geom, dict) and "name" in geom:
            cf_attrs["grid_mapping"] = geom["name"]

        dim_names: list[str] = []
        shape: list[int] = []
        chunks: list[int] = []

        for dim_ref in var.get("dims", []):
            if not isinstance(dim_ref, dict):
                # Non-dict dim references (e.g. unresolved anchors) are skipped
                continue
            dname = dim_ref.get("name", "")
            dvalues = _eval_values(dim_ref.get("values", []))
            chunk = dim_ref.get("chunk", len(dvalues))
            dim_names.append(dname)
            shape.append(len(dvalues))
            chunks.append(chunk)

        if not shape:
            continue

        data = da.empty(shape, chunks=tuple(chunks))
        data_vars[var_name] = xr.DataArray(data, dims=dim_names, attrs=cf_attrs)

    return xr.Dataset(data_vars, coords=coords)
