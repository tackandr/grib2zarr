# ADR 0009: GRIB2 and CF Metadata Model

- Status: Proposed  
- Date: 2026-11-11  
- Related: ADR-0007 — Declarative Configuration Architecture for GRIB-to-Zarr Conversion

## Context

The GRIB-to-Zarr converter transforms meteorological data represented in GRIB2 messages into multidimensional Zarr datasets following Climate and Forecast (CF) metadata conventions.

GRIB2 and CF describe meteorological data using different metadata models.

GRIB2 metadata is organized into sections and templates describing parameters, forecast times, vertical levels, grid definitions, and other characteristics of the encoded messages.

CF conventions describe multidimensional datasets using standardized coordinate variables, attributes, grid mappings, and relationships between variables.

The converter must work with metadata from both systems.

For example, a GRIB2 parameter may be identified using:

```yaml
grib2:
  discipline: 0
  parameterCategory: 2
  parameterNumber: 1
```

The corresponding output variable may be described using CF attributes:

```yaml
cf:
  standard_name: wind_speed
  units: m s-1
```

Spatial geometry similarly has GRIB2 and CF representations:

```yaml
grib2:
  gridDefinitionTemplateNumber: 30

cf:
  crs:
    grid_mapping_name: lambert_conformal_conic
    standard_parallel: [63.3, 63.3]
    longitude_of_central_meridian: 15.0
    latitude_of_projection_origin: 63.3
```

The two metadata systems serve different purposes and cannot be treated as interchangeable.

The configuration architecture must define how these metadata systems coexist, which components interpret them, and how metadata is represented internally.

## Decision

The configuration will represent **GRIB2 and CF metadata in separate mappings**, named `grib2` and `cf`.

These mappings may appear in coordinate, geometry, and variable definitions.

GRIB2 metadata describes the source GRIB representation and provides information used to identify or interpret GRIB messages.

CF metadata describes the intended representation of coordinates, geometries, and variables in the output dataset.

The parser will preserve metadata without performing implicit translations between GRIB2 and CF attributes.

Metadata interpretation, validation, and conversion will be handled by the appropriate downstream components.

### 1. Separation of metadata namespaces

GRIB2 metadata will be contained in a `grib2` mapping.

CF metadata will be contained in a `cf` mapping.

Example:

```yaml
variables:
  - name: wind_speed
    grib2:
      discipline: 0
      parameterCategory: 2
      parameterNumber: 1
    cf:
      standard_name: wind_speed
      units: m s-1
```

The same property name may appear in different metadata namespaces without implying identical semantics.

The parser must preserve the distinction between the two mappings.

GRIB2 metadata must not be automatically converted into CF metadata during configuration parsing.

Likewise, CF metadata must not be used to infer missing GRIB2 metadata during parsing.

Any mapping between GRIB2 and CF metadata belongs to the conversion logic or a dedicated metadata transformation component.

### 2. Metadata associated with coordinates

Coordinates may contain both GRIB2 and CF metadata.

Example:

```yaml
coordinates:
  - hybrid: &hybrid
      name: hybrid
      grib2:
        typeOfFirstFixedSurface: 105
      cf:
        axis: Z
        standard_name: atmosphere_hybrid_sigma_pressure_coordinate
        positive: down
      values: [1, 2, 3]
      chunk: 1
```

The `grib2` mapping describes GRIB2 characteristics associated with the coordinate.

The `cf` mapping describes how the coordinate is represented in the output dataset.

Coordinate metadata may include information about:

- Temporal coordinates.
- Vertical coordinates.
- Horizontal coordinates.
- Units.
- Coordinate direction.
- Coordinate standard names.

The exact set of supported attributes depends on the coordinate type and is defined in the configuration schema.

A coordinate is not required to contain both metadata mappings unless specifically required by its schema.

### 3. Metadata associated with geometries

The dataset's single geometry may contain GRIB2 grid metadata and CF spatial metadata.

Example:

```yaml
geometry:
  projection: &projection
    name: projection
    grib2:
      gridDefinitionTemplateNumber: 30
    cf:
      crs:
        grid_mapping_name: lambert_conformal_conic
        standard_parallel: [63.3, 63.3]
        longitude_of_central_meridian: 15.0
        latitude_of_projection_origin: 63.3
      coords:
        - *y
        - *x
```

The `grib2` mapping identifies or describes the grid representation used in the source data.

The `cf` mapping describes the spatial reference system and its associated coordinates.

The `cf` mapping of a geometry contains two distinct elements:

- `crs`: CF grid mapping attributes describing the coordinate reference system.
- `coords`: References to the coordinate definitions associated with the geometry.

Although `coords` is located within the `cf` mapping, it is a configuration-level relationship rather than a CF attribute that should be copied directly into the output dataset.

The parser must preserve this distinction.

The conversion component is responsible for translating the geometry definition into the appropriate CF grid mapping representation in the output dataset.

### 4. Metadata associated with variables

Variables may contain GRIB2 parameter metadata and CF variable attributes.

Example:

```yaml
variables:
  - name: wind_speed
    grib2:
      discipline: 0
      parameterCategory: 2
      parameterNumber: 1
    cf:
      standard_name: wind_speed
      units: m s-1
    geometry: *projection
    dims:
      - *steps
      - *hybrid
      - *y
      - *x
```

The `grib2` mapping identifies the corresponding source parameter.

The `cf` mapping describes the resulting output variable.

The variable's geometry and dimensions are represented separately from its metadata mappings.

The parser must not infer the variable's dimensions or geometry from GRIB2 or CF metadata.

These relationships are explicitly declared in the configuration.

### 5. GRIB2 metadata representation

GRIB2 metadata will be represented as mappings of GRIB2 metadata keys to configuration values.

The configuration uses metadata key names corresponding to the GRIB-reading interface.

For example:

```yaml
grib2:
  discipline: 0
  parameterCategory: 2
  parameterNumber: 1
```

The parser must preserve the key names and values.

GRIB2 metadata values may include scalar values and sequences where supported by the configuration schema and GRIB-reading interface.

The parser must not assume that every GRIB2 metadata key is an integer.

The GRIB-reading component is responsible for interpreting metadata according to GRIB2 conventions and the capabilities of the GRIB decoder.

#### GRIB2 metadata roles

GRIB2 metadata can serve different purposes depending on where it appears.

| Location | Primary role |
|---|---|
| Coordinate | Identify or describe GRIB2 coordinate characteristics |
| Geometry | Identify or describe the GRIB2 grid |
| Variable | Identify or describe the GRIB2 parameter |

The converter may use metadata from multiple definitions when selecting and interpreting GRIB messages.

For example, identifying a specific field may require information about its parameter, vertical level, and grid.

The precise rules for combining GRIB2 metadata across configuration entities belong to the GRIB message selection specification.

This ADR does not define whether GRIB2 metadata is interpreted as an exact match, a partial match, or another selection criterion.

### 6. CF metadata representation

CF metadata will be represented using mappings corresponding to CF attributes and configuration-specific CF structures.

Example:

```yaml
cf:
  standard_name: wind_speed
  units: m s-1
```

The parser must preserve CF attribute names and values.

CF metadata can include scalar attributes and array-valued attributes.

For example:

```yaml
cf:
  crs:
    grid_mapping_name: lambert_conformal_conic
    standard_parallel: [63.3, 63.3]
    longitude_of_central_meridian: 15.0
    latitude_of_projection_origin: 63.3
```

The parser must not assume that all CF metadata values are strings.

The exact permitted types and required attributes are defined by the configuration schema and applicable CF conventions.

#### CF attributes and configuration structures

Not every field inside a `cf` mapping represents an attribute to be written directly to the output dataset.

For example:

```yaml
cf:
  crs:
    grid_mapping_name: lambert_conformal_conic
  coords:
    - *y
    - *x
```

The `crs` mapping contains CF grid mapping attributes.

The `coords` sequence contains references to configuration coordinate definitions.

The conversion component must interpret these structures and generate the appropriate output representation.

It must not blindly copy the entire `cf` mapping as Zarr attributes.

### 7. Metadata preservation

The parser must preserve supported metadata attributes without unnecessary transformations.

In particular:

- Metadata key names must retain their original spelling and case.
- Numeric values must remain numeric.
- String values must remain strings.
- Boolean values must remain boolean.
- Sequence ordering must be preserved.
- Nested mappings must retain their structure where permitted.

The parser must not silently rename metadata keys, change units, or substitute values.

Any metadata transformation required for the output dataset must be performed explicitly by the conversion component.

### 8. Metadata extensibility

GRIB2 and CF metadata standards contain more attributes than those shown in the example configuration.

The configuration format must support additional metadata without requiring changes to the top-level architecture.

The `grib2` and `cf` mappings therefore act as extensible metadata containers.

However, extensibility does not imply that arbitrary values are always valid.

The configuration schema defines required attributes, permitted value types, and any restrictions on supported metadata.

The parser must distinguish between:

- Structural properties defined by the configuration format.
- Metadata attributes supported by the converter.
- Unsupported or invalid metadata.

The handling of unknown metadata keys is defined by the configuration schema and validation policy.

The parser must not silently discard unknown metadata.

### 9. Metadata validation responsibilities

Metadata validation will be separated into two levels.

#### Structural validation

Structural validation is performed by the configuration parser.

It includes checking:

- Whether metadata mappings have the expected structure.
- Whether required metadata fields are present.
- Whether values have supported YAML types.
- Whether configuration-specific structures such as `cf.crs` and `cf.coords` are valid.

#### Semantic validation

Semantic validation checks whether metadata is meaningful according to GRIB2 and CF conventions.

Examples include:

- Whether a GRIB2 parameter identifier is valid.
- Whether a CF `standard_name` is recognized.
- Whether the units are compatible with the specified standard name.
- Whether a grid mapping contains the required projection attributes.
- Whether a vertical coordinate contains the required metadata for its coordinate transformation.

Semantic validation may be performed by dedicated validation components or during conversion.

Successful structural validation does not guarantee that the resulting dataset will be fully CF-compliant.

The converter remains responsible for generating an output dataset satisfying the applicable conventions.

### 10. Normalized metadata representation

The parser will produce a normalized configuration model in which GRIB2 and CF metadata remain logically separate.

For a variable, the representation should be conceptually equivalent to:

```json
{
  "name": "wind_speed",
  "grib2": {
    "discipline": 0,
    "parameterCategory": 2,
    "parameterNumber": 1
  },
  "cf": {
    "standard_name": "wind_speed",
    "units": "m s-1"
  },
  "geometry": "projection",
  "dims": [
    "steps",
    "hybrid",
    "y",
    "x"
  ]
}
```

The internal model may use typed objects rather than generic dictionaries.

However, the separation between GRIB2 metadata, CF metadata, and configuration-specific relationships must be preserved.

## Alternatives considered

### 1. Strictly enumerated metadata fields

The configuration schema could allow only a fixed list of GRIB2 and CF attributes.

**Advantages:**

- Strong validation.
- Easier documentation of supported fields.
- Unknown attributes could be rejected immediately.

**Disadvantages:**

- Supporting additional GRIB2 keys or CF attributes would require frequent schema updates.
- The format would be less adaptable to new datasets and metadata requirements.
- The parser would become more tightly coupled to specific GRIB2 templates and CF features.

**Not selected as a general architectural restriction:** Metadata containers will be extensible, while the configuration schema and validation components define supported attributes and constraints.

## Consequences

### Positive

- GRIB2 source metadata and CF output metadata have clearly separated responsibilities.
- Metadata can be associated with coordinates, geometries, and variables.
- The configuration can express metadata required for different GRIB2 parameters, vertical coordinates, and spatial grids.
- CF output metadata is explicitly controlled by the configuration.
- Metadata can be extended without changing the overall configuration architecture.
- The parser can preserve metadata without implementing GRIB2 decoding or CF transformations.
- Structural validation and standards-level semantic validation can evolve independently.

### Negative

- Some information may be represented in both GRIB2 and CF metadata.
- The configuration may contain inconsistent metadata across the two namespaces.
- Metadata extensibility requires a clear validation policy.
- The converter must interpret configuration-specific structures inside CF mappings rather than copying them directly as attributes.
- Explicit CF metadata requires configuration authors to understand the intended output representation.
- Structural parsing alone cannot guarantee CF compliance.

## Scope

This ADR defines:

- The separation of GRIB2 and CF metadata.
- The placement of metadata within configuration entities.
- The general representation of metadata mappings.
- Metadata preservation principles.
- Metadata extensibility.
- The separation of structural and semantic validation responsibilities.

It does not define:

- The complete list of supported GRIB2 metadata keys.
- The complete list of supported CF attributes.
- GRIB2 message selection and matching rules.
- GRIB2-to-CF metadata conversion algorithms.
- Projection transformations.
- Hybrid vertical coordinate transformations.
- Time coordinate calculations.
- Detailed parser validation behavior.
- Zarr metadata serialization.

These concerns are addressed by the configuration schema, dedicated conversion components, and subsequent ADRs.

## Related documents

- **ADR-0007:** Declarative Configuration Architecture for GRIB-to-Zarr Conversion
- **ADR-0008:** Configuration References and Identity
- **ADR-00010:** Configuration Parsing and Validation *(planned)*
- **Configuration Schema:** Detailed YAML structure and field definitions *(planned)*
