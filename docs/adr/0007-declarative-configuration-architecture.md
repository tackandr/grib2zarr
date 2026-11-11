# ADR 0007: Declarative Configuration Architecture for GRIB-to-Zarr Conversion

- Status: Proposed  
- Date: 2026-11-11

## Context

The GRIB-to-Zarr converter transforms meteorological data stored in GRIB files into Zarr datasets following the Climate and Forecast (CF) metadata conventions.

GRIB and Zarr represent meteorological data differently.

GRIB organizes data into messages containing parameter identification, temporal information, vertical level information, and grid definitions. Zarr organizes data into multidimensional arrays with explicit dimensions, coordinates, and attributes.

The converter therefore needs information describing how GRIB messages are represented in the resulting Zarr dataset.

Different datasets may require different coordinate systems, dimensions, variables, metadata, and chunking strategies. These differences should be configurable without modifying the converter's source code.

A configuration format is needed to describe the target dataset and provide the metadata required to map GRIB messages into its structure.

The configuration must support:

- Defining coordinates and their values.
- Defining spatial geometries and coordinate reference systems.
- Defining variables and their dimensions.
- Associating variables with GRIB2 parameter metadata.
- Describing output metadata following CF conventions.
- Specifying chunk sizes for Zarr storage.
- Reusing coordinate and geometry definitions across multiple variables.

The configuration should be human-readable, declarative, and suitable for version control.

## Decision

The GRIB-to-Zarr converter will use a **declarative YAML configuration file** to describe the structure and metadata of the target Zarr dataset.

The configuration will consist of three top-level sections:

```yaml
coordinates: []
geometries: []
variables: []
```

Each section contains a sequence of definitions.

### Coordinates

The `coordinates` section defines the coordinate axes used by the output dataset.

A coordinate definition may contain:

- A unique name.
- Explicit coordinate values or a function that produces such values.
- CF coordinate metadata.
- GRIB2 metadata associated with the coordinate.
- Reference time information.
- A Zarr chunk size.

Coordinates describe the dimensions along which variable data is organized.

Coordinate definitions can be reused by multiple variables and geometries.

Example:

```yaml
coordinates:
  - x: &x
      name: x
      cf:
        axis: X
        standard_name: projection_x_coordinate
        units: m
      values: [0, 2500, 5000]
      chunk: 3
```

### Geometries

The `geometries` section defines the spatial geometry associated with variables.

A geometry definition may contain:

- A unique name.
- GRIB2 grid identification metadata.
- A coordinate reference system described using CF grid mapping attributes.
- References to the spatial coordinates defining the geometry.

Geometry definitions can be shared by multiple variables.

Example:

```yaml
geometries:
  - projection: &projection
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

### Variables

The `variables` section defines the data variables to be produced in the Zarr dataset.

A variable definition may contain:

- A unique name.
- GRIB2 metadata identifying the source parameter.
- CF metadata describing the output variable.
- A reference to its spatial geometry.
- An ordered list of coordinate dimensions.

Example:

```yaml
variables:
  - name: ws
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

The order of dimensions determines the dimension order of the resulting variable.

Coordinates and geometries are defined independently of variables to allow reuse and ensure consistent definitions across the dataset.

## Design principles

The configuration architecture follows these principles.

### 1. Declarative dataset description

The configuration describes **what the resulting dataset should contain**, rather than the sequence of operations required to create it.

It specifies the intended dimensions, coordinates, variables, metadata, and storage chunking.

The converter determines how to process GRIB messages to produce the specified dataset.

### 2. Separation of definitions

Coordinates, geometries, and variables are modeled as separate entities because they represent different concepts.

- Coordinates describe dimension values and their metadata.
- Geometries describe spatial reference systems and spatial coordinate relationships.
- Variables describe meteorological fields and their multidimensional organization.

This separation allows the same coordinates and geometries to be shared by multiple variables without duplicating their definitions.

### 3. Explicit dataset structure

The configuration explicitly defines the dimensions and coordinate values of the target dataset.

The converter must not rely solely on the structure of individual GRIB messages to determine the output dataset layout.

This makes the intended output structure predictable and allows it to be validated independently of input data.

For a variable with dimensions:

```yaml
dims:
  - *steps
  - *hybrid
  - *y
  - *x
```

the output dimension order is:

```text
(steps, hybrid, y, x)
```

The dimension lengths are determined by the corresponding coordinate definitions.

### 4. Separation of GRIB2 and CF metadata

The configuration distinguishes between metadata describing the GRIB2 source data and metadata describing the CF-compliant output dataset.

GRIB2 metadata is contained in a `grib2` mapping.

CF metadata is contained in a `cf` mapping.

This distinction reflects their different responsibilities:

- GRIB2 metadata identifies or describes the source messages and grids.
- CF metadata describes the coordinates, variables, and spatial reference systems of the output dataset.

The detailed interpretation and validation of these metadata mappings are defined separately.

### 5. Reusable definitions

Coordinates and geometries are defined independently and can be referenced from other configuration sections.

The configuration uses YAML anchors and aliases to express these relationships.

For example, a coordinate may be defined once and referenced by several variables.

The precise reference semantics, identity rules, and normalization strategy are addressed in a separate ADR.

### 6. Separation of configuration and conversion logic

The configuration format is independent of the implementation of GRIB decoding and Zarr writing.

A configuration parser is responsible for interpreting the configuration and providing a structured representation to the converter.

The converter is responsible for processing GRIB messages and writing the resulting Zarr dataset.

The configuration does not prescribe which GRIB decoding library, Zarr implementation, or programming language must be used.

### 7. Human-readable and version-controlled

YAML is selected because it supports:

- Human-readable structured data.
- Nested mappings and sequences.
- Reusable definitions through anchors and aliases.
- Standard parsing libraries.
- Reviewable changes in version control.

Configuration files are intended to be maintained alongside the code or dataset definitions that use them.

## Alternatives considered

### 1. Automatic dataset structure inference from GRIB

The converter could derive dimensions, coordinates, geometries, and variables directly from the input GRIB messages.

**Not selected as the primary configuration approach because:**

- The resulting dataset structure would depend on the contents of the input files.
- Missing GRIB messages could affect the inferred dataset structure.
- Output dimension ordering and chunking would require additional rules.
- Ensuring consistent output across different GRIB inputs would be more difficult.
- Mapping of GRIB metadata to CF might not be defined for non-standard parameters

GRIB metadata may still be used to validate or populate the configured dataset.

## Consequences

### Positive

- Dataset structure can be changed without modifying converter code.
- The intended output dimensions and coordinate values are explicitly defined.
- Coordinates and geometries can be reused across variables.
- GRIB2 and CF metadata are clearly separated.
- Dataset definitions can be reviewed and version-controlled.
- Configuration parsing and validation can be implemented independently of GRIB decoding.
- The same converter can support different datasets using different configuration files.

### Negative

- Explicitly defining coordinate values may produce large configuration files for large or irregular datasets.
- Configuration authors must understand the intended dataset structure.
- References between definitions introduce a need for reference resolution and validation.
- Some information may appear in both GRIB messages and the configuration, requiring consistency checks.
- The declarative format may need extensions to support datasets whose structure cannot be described using explicit coordinate definitions.

## Scope

This ADR establishes the high-level configuration architecture and its three principal entities.

It does not define:

- The complete configuration schema or individual field requirements.
- The detailed semantics of YAML anchors and aliases.
- Reference resolution and object identity rules.
- GRIB2 parameter matching behavior.
- CF metadata validation or coordinate transformation rules.
- Parser validation and error-reporting requirements.
- Zarr writing behavior.
- The internal programming-language representation of the configuration.

These concerns are addressed by the configuration schema, subsequent ADRs, and the converter implementation.

## Related documents

- **ADR-0008:** Configuration References and Identity *(planned)*
- **ADR-0009:** GRIB2 and CF Metadata Model *(planned)*
- **ADR-0010:** Configuration Parsing and Validation *(planned)*
- **Configuration Schema:** Detailed YAML structure and field definitions *(planned)*
