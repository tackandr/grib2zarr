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

from datetime import datetime, timedelta

import pyproj
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


def _build_datetime_values(reference_time: str, step_values: list) -> np.ndarray:
    """Convert step offsets (hours) to a ``numpy.datetime64`` array.

    Parameters
    ----------
    reference_time:
        ISO 8601 reference datetime string, e.g. ``"2026-03-27T09:00:00"``.
    step_values:
        List of step offsets in hours from the reference time.

    Returns
    -------
    numpy.ndarray
        Array of ``numpy.datetime64`` values with nanosecond precision.
    """
    ref_dt = datetime.fromisoformat(reference_time)
    dt_values = [ref_dt + timedelta(hours=float(s)) for s in step_values]
    return np.array(dt_values, dtype="datetime64[ns]")


def _build_latlon_from_crs(
    crs_attrs: dict,
    x_name: str,
    y_name: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[xr.Variable, xr.Variable]:
    """Compute 2-D latitude/longitude arrays from projected x/y coordinates.

    Uses the CF-convention CRS attributes (as stored on the grid-mapping
    variable) to build a :class:`pyproj.Transformer` from the projected
    coordinate system to WGS 84 geographic coordinates, then evaluates the
    full ``(ny, nx)`` meshgrid.

    Parameters
    ----------
    crs_attrs:
        CF grid-mapping attributes dict (e.g. produced by
        ``config["geometries"][…]["cf"]["crs"]``).
    x_name:
        Name of the x (``projection_x_coordinate``) dimension.
    y_name:
        Name of the y (``projection_y_coordinate``) dimension.
    x_values:
        1-D array of x coordinate values in the projected CRS (metres).
    y_values:
        1-D array of y coordinate values in the projected CRS (metres).

    Returns
    -------
    tuple[xr.Variable, xr.Variable]
        2-D ``(y_name, x_name)`` *latitude* and *longitude* variables
        with CF-standard attributes.
    """
    crs = pyproj.CRS.from_cf(crs_attrs)
    transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(x_values, y_values)
    lon, lat = transformer.transform(xx, yy)
    dims = (y_name, x_name)
    lat_var = xr.Variable(
        dims,
        lat,
        attrs={
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
        },
    )
    lon_var = xr.Variable(
        dims,
        lon,
        attrs={
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
        },
    )
    return lat_var, lon_var


def build_dataset(config: dict) -> xr.Dataset:
    """Build an empty xarray Dataset from a parsed YAML configuration.

    The returned dataset contains:

    * **Coordinates** – one per entry in ``config['coordinates']``, with CF
      attributes.  Auxiliary parameter variables (e.g. *a* and *b*
      coefficients for a hybrid sigma-pressure coordinate) are added as
      non-index coordinates.
    * **Grid-mapping variables** – one scalar integer variable per entry in
      ``config['geometries']``, carrying CRS metadata as attributes following
      the CF grid_mapping convention.  If the geometry references spatial
      coordinate axes (``axis: Y`` and ``axis: X``), 2-D ``latitude`` and
      ``longitude`` coordinate variables are also added (derived via
      :func:`_build_latlon_from_crs`).
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
            ref_time = info.get("reference_time")
            if ref_time is not None:
                step_vals = _eval_values(info["values"])
                values = _build_datetime_values(ref_time, step_vals)
            else:
                values = np.array(_eval_values(info["values"]), dtype=float)
            cf_attrs = dict(info.get("cf", {}))
            if ref_time is not None:
                # xarray manages the ``units`` encoding for datetime64 arrays
                # automatically.  Keeping a user-supplied ``units`` attr would
                # cause a "Key already exists" conflict during to_zarr encoding.
                cf_attrs.pop("units", None)
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

            # Derive 2-D latitude/longitude from the projected x/y axes when
            # the geometry references both a Y- and an X-axis coordinate.
            if crs_attrs:
                x_name = y_name = None
                geom_cf = info.get("cf", {})
                for coord_ref in geom_cf.get("coords", []):
                    if not isinstance(coord_ref, dict):
                        continue
                    coord_cf_attrs = coord_ref.get("cf", {})
                    axis = coord_cf_attrs.get("axis", "")
                    cname = coord_ref.get("name", "")
                    if axis == "X":
                        x_name = cname
                    elif axis == "Y":
                        y_name = cname
                if x_name and y_name and x_name in coords and y_name in coords:
                    x_vals = coords[x_name].values
                    y_vals = coords[y_name].values
                    if x_vals.ndim == 1 and y_vals.ndim == 1:
                        lat_var, lon_var = _build_latlon_from_crs(
                            crs_attrs,
                            x_name,
                            y_name,
                            x_vals,
                            y_vals,
                        )
                        if "latitude" not in coords:
                            coords["latitude"] = lat_var
                        if "longitude" not in coords:
                            coords["longitude"] = lon_var

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

        data = da.empty(shape, chunks=tuple(chunks), dtype=np.float32)
        data_vars[var_name] = xr.DataArray(data, dims=dim_names, attrs=cf_attrs)

    return xr.Dataset(data_vars, coords=coords)
