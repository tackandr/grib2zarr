# GRIB-to-Zarr Configuration Schema

## 1. Overview

This document specifies the YAML configuration format used by the GRIB-to-Zarr converter.

The configuration describes the structure and metadata of the target Zarr dataset and provides the information required to select GRIB2 messages to populate its variables.

The configuration defines:

- Coordinates and their values.
- Spatial geometries and coordinate reference systems.
- Variables and their dimensions.
- GRIB2 metadata used for message identification.
- CF metadata describing the output dataset.
- Zarr chunk sizes.
- Relationships between coordinates, geometries, and variables.

The configuration is declarative. It specifies the intended dataset structure rather than the operations required to construct it.

The parser must validate the configuration and produce a normalized representation before GRIB data processing begins.

### 1.1 Related architecture decisions

This specification implements the following architectural decisions:

- **ADR-0007:** Declarative Configuration Architecture
- **ADR-0008:** Configuration References and Identity
- **ADR-0009:** GRIB2 and CF Metadata Model
- **ADR-0010:** Configuration Parsing and Validation

### 1.2 Terminology

| Term | Definition |
|---|---|
| Coordinate | A named dimension with an ordered sequence of values |
| Geometry | A spatial reference system associated with spatial coordinates |
| Variable | An output data array with dimensions, metadata, and a geometry |
| Reference | A YAML alias pointing to a declared coordinate or geometry |
| Spatial dimensions | Dimensions defining the horizontal field represented by a GRIB2 message |
| Message dimensions | Variable dimensions that distinguish individual GRIB2 messages |
| Normalized configuration | Parsed configuration with evaluated coordinate values and resolved references |
| Message selector | Criteria used to identify GRIB2 messages |
| Message assignment | Association between a GRIB2 message and an output array position |

---

## 2. Root structure

A configuration is a YAML document containing three top-level sections:

```yaml
coordinates: []
geometry: {}
variables: []
```

### 2.1 Root properties

| Property | Type | Required | Description |
|---|---|---|---|
| `coordinates` | sequence | Yes | Coordinate definitions |
| `geometry` | mapping | Yes | Single geometry definition |
| `variables` | sequence | Yes | Variable definitions |

### 2.2 Constraints

- The root must be a YAML mapping.
- All three sections must exist.
- `coordinates` and `variables` must contain YAML sequences.
- `geometry` must contain a YAML mapping with exactly one entry.
- At least one variable must be defined.
- Unknown top-level properties are not permitted.
- Mapping keys are case-sensitive.
- Duplicate YAML mapping keys are not permitted.

The configuration uses YAML anchors and aliases to reference definitions.

Because YAML aliases must refer to previously defined anchors, the conventional section order is:

1. `coordinates`
2. `geometry`
3. `variables`

The parser must not depend on the textual ordering of top-level mapping keys beyond YAML's alias-resolution requirements.

---

## 3. Coordinates

The `coordinates` section defines the coordinate dimensions used by the target dataset.

Each coordinate has a name, an ordered sequence of values, and a Zarr chunk size.

Coordinates may also contain GRIB2 and CF metadata.

### 3.1 Structure

```yaml
coordinates:
  - steps: &steps
      name: steps
      reference_time: "2026-03-27T09:00:00"
      cf:
        axis: T
        standard_name: time
      values: [1, 2, 3]
      chunk: 1
```

Each sequence item must contain exactly one mapping entry.

The outer mapping key identifies the coordinate definition.

The mapping value contains the coordinate properties.

### 3.2 Properties

| Property | Type | Required | Description |
|---|---|---|---|
| Outer mapping key | string | Yes | Coordinate identifier |
| `name` | string | Yes | Coordinate name |
| `reference_time` | string | No | Reference time associated with the coordinate |
| `grib2` | mapping | No | GRIB2 coordinate metadata |
| `cf` | mapping | No | CF coordinate attributes |
| `values` | sequence or expression string | Yes | Coordinate values |
| `chunk` | positive integer | Yes | Zarr chunk length |

Unknown properties outside the `grib2` and `cf` mappings are not permitted.

### 3.3 Coordinate identity

The outer mapping key must match the `name` property.

Valid:

```yaml
- hybrid: &hybrid
    name: hybrid
    values: [1, 2, 3]
    chunk: 1
```

Invalid:

```yaml
- hybrid: &hybrid
    name: levels
    values: [1, 2, 3]
    chunk: 1
```

Coordinate names must be unique within the `coordinates` section.

The YAML anchor name does not determine coordinate identity.

For example:

```yaml
- hybrid: &model_levels
    name: hybrid
    values: [1, 2, 3]
    chunk: 1
```

defines the coordinate `hybrid`.

### 3.4 Coordinate values

The `values` property defines the ordered values of a coordinate.

Two representations are supported:

1. Explicit YAML sequences.
2. Expression strings.

Both representations must produce the same normalized coordinate representation.

#### Explicit values

```yaml
values: [1, 2, 3, 4, 5]
```

Explicit values may contain numeric or string values.

Values must satisfy the following constraints:

- The sequence must not be empty.
- Values must be scalar.
- Values must be unique.
- Numeric values must be finite.
- Boolean values are not supported.
- Nested sequences and mappings are not supported.

The parser must preserve value order.

Coordinate values are not required to be monotonically increasing.

#### Expression-based values

```yaml
values: "range(1, 6)"
```

or:

```yaml
values: "[x for x in range(1, 6)]"
```

Expressions are evaluated during configuration parsing.

Both examples produce:

```python
[1, 2, 3, 4, 5]
```

Expression evaluation is defined in Section 7.

### 3.5 Coordinate length

The coordinate length is derived from its normalized values.

For example:

```yaml
values: [0, 2500, 5000]
```

produces:

```text
length = 3
```

For generated values:

```yaml
values: "range(1, 66)"
```

the length is:

```text
length = 65
```

Coordinate length is not configured separately.

### 3.6 Chunk size

The `chunk` property specifies the Zarr chunk length along the coordinate dimension.

Example:

```yaml
chunk: 1
```

Constraints:

- Must be an integer.
- Must be greater than zero.
- Boolean values are not accepted.
- May exceed the coordinate length.

The chunk size is used when deriving the chunk shape of variables referencing the coordinate.

For example:

```text
dimensions = [steps, hybrid, y, x]

chunks = [1, 1, 3, 3]
```

### 3.7 Reference time

A coordinate may define a reference time.

Example:

```yaml
reference_time: "2026-03-27T09:00:00"
```

The property must be a string containing a supported ISO 8601 date-time representation.

The parser must validate its syntax.

The parser must not interpret the coordinate values as Unix timestamps or automatically convert them into absolute timestamps.

The relationship between reference time, coordinate values, and GRIB2 forecast-time metadata is determined by the GRIB message selection implementation.

Configuration authors should quote date-time strings to avoid implicit YAML timestamp conversion.

### 3.8 GRIB2 coordinate metadata

Coordinates may contain a `grib2` mapping.

Example:

```yaml
grib2:
  typeOfFirstFixedSurface: 105
```

The mapping contains GRIB2 metadata associated with the coordinate.

The parser must preserve metadata key names and values.

The converter uses this metadata when constructing or validating GRIB2 message selection criteria.

The precise relationship between coordinate values and GRIB2 keys is defined by the GRIB selector implementation.

For example, the schema does not independently establish whether hybrid coordinate values correspond to the GRIB2 key `level` or another key.

### 3.9 CF coordinate metadata

Coordinates may contain a `cf` mapping.

Example:

```yaml
cf:
  axis: Z
  standard_name: atmosphere_hybrid_sigma_pressure_coordinate
  positive: down
```

Common CF coordinate attributes include:

| Attribute | Type | Description |
|---|---|---|
| `axis` | string | Coordinate axis designation |
| `standard_name` | string | CF standard name |
| `long_name` | string | Descriptive coordinate name |
| `units` | string | Coordinate units |
| `positive` | string | Direction of increasing vertical coordinate |
| `calendar` | string | Calendar used for time coordinates |

Targeted validation rules:

- `axis`, when present, must be `T`, `Z`, `Y`, or `X`.
- `positive`, when present, must be `up` or `down`.
- `standard_name`, when present, must be a string.
- `units`, when present, must be a string.

The parser does not verify all CF standard-name or unit compatibility requirements.

---

## 4. Geometry

The `geometry` section defines the single spatial geometry of the dataset.

All spatial variables use this geometry, consistent with ADR-0001. A geometry describes a coordinate reference system and references the spatial coordinates defining its grid.

### 4.1 Structure

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

The `geometry` mapping contains a single key, which is the geometry identifier.

### 4.2 Properties

| Property | Type | Required | Description |
|---|---|---|---|
| Outer mapping key | string | Yes | Geometry identifier |
| `name` | string | Yes | Geometry name |
| `grib2` | mapping | No | GRIB2 grid metadata |
| `cf` | mapping | Yes | CF geometry definition |
| `cf.crs` | mapping | Yes | Coordinate reference system |
| `cf.coords` | sequence of references | Yes | Spatial coordinates |

Unknown properties outside the `grib2` and `cf` mappings are not permitted.

### 4.3 Geometry identity

The outer mapping key must match the `name` property.

The `geometry` section contains exactly one geometry definition.

The YAML anchor name is not part of geometry identity.

### 4.4 GRIB2 geometry metadata

The `grib2` mapping describes GRIB2 grid characteristics.

Example:

```yaml
grib2:
  gridDefinitionTemplateNumber: 30
```

The parser must preserve the metadata.

The GRIB-reading component is responsible for interpreting the grid definition and verifying compatibility with the configured geometry.

### 4.5 Coordinate reference system

The `cf.crs` mapping contains CF grid mapping attributes.

Example:

```yaml
crs:
  grid_mapping_name: lambert_conformal_conic
  standard_parallel: [63.3, 63.3]
  longitude_of_central_meridian: 15.0
  latitude_of_projection_origin: 63.3
```

The following constraints apply:

- `crs` must be a mapping.
- `grid_mapping_name` is required.
- `grid_mapping_name` must be a non-empty string.
- Additional CRS properties may contain supported metadata values.
- Numeric values must be finite.
- Sequences must preserve their order.

The parser does not perform mathematical projection validation.

The converter is responsible for constructing the corresponding CF grid mapping representation in the output dataset.

### 4.6 Spatial coordinate references

The `cf.coords` property identifies the coordinates associated with the geometry.

Example:

```yaml
coords:
  - *y
  - *x
```

Each entry must reference a coordinate declared in the top-level `coordinates` section.

Constraints:

- Must be a non-empty sequence.
- Every entry must reference a declared coordinate.
- Duplicate coordinate references are not permitted.
- Reference order must be preserved.

The order of these coordinates defines the spatial dimension order used by the geometry.

For example:

```text
spatial_dimensions = [y, x]
```

The geometry's spatial dimensions are used to determine which variable dimensions represent the horizontal field supplied by each GRIB2 message.

### 4.7 Geometry compatibility

A variable using a geometry must contain the geometry's spatial coordinates among its dimensions.

The relative ordering of these coordinates must match the geometry definition.

For example, a geometry declaring:

```yaml
coords:
  - *y
  - *x
```

is compatible with:

```yaml
dims:
  - *steps
  - *hybrid
  - *y
  - *x
```

It is not compatible with:

```yaml
dims:
  - *steps
  - *hybrid
  - *x
  - *y
```

The parser validates the relationship between variable dimensions and geometry coordinates.

Compatibility with actual GRIB2 grids is validated during conversion.

---

## 5. Variables

The `variables` section defines the data variables produced in the target Zarr dataset.

Each variable identifies its GRIB2 source parameter, CF metadata, geometry, and dimensions.

### 5.1 Structure

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

Unlike coordinate and geometry definitions, variables are not wrapped in a mapping keyed by their name.

### 5.2 Properties

| Property | Type | Required | Description |
|---|---|---|---|
| `name` | string | Yes | Output variable name |
| `grib2` | mapping | Yes | GRIB2 parameter metadata |
| `cf` | mapping | No | CF variable attributes |
| `geometry` | geometry reference | Yes | Spatial geometry |
| `dims` | sequence of coordinate references | Yes | Ordered variable dimensions |

Unknown properties outside the `grib2` and `cf` mappings are not permitted.

### 5.3 Variable identity

The `name` property identifies the variable in the output dataset.

Variable names must be unique within the `variables` section.

### 5.4 GRIB2 parameter metadata

The `grib2` mapping defines GRIB2 metadata associated with the variable.

The following properties are required:

| Property | Type | Description |
|---|---|---|
| `discipline` | non-negative integer | GRIB2 discipline |
| `parameterCategory` | non-negative integer | Parameter category |
| `parameterNumber` | non-negative integer | Parameter number |

Example:

```yaml
grib2:
  discipline: 0
  parameterCategory: 2
  parameterNumber: 1
```

Additional GRIB2 metadata properties are permitted.

The parser performs structural and basic numeric validation.

It does not validate every GRIB2 parameter combination against the complete GRIB2 code tables.

The GRIB-reading component interprets these properties when selecting input messages.

### 5.5 CF variable metadata

The `cf` mapping describes the output variable.

Example:

```yaml
cf:
  standard_name: wind_speed
  units: m s-1
```

Common properties include:

| Attribute | Type | Description |
|---|---|---|
| `standard_name` | string | CF standard name |
| `long_name` | string | Descriptive variable name |
| `units` | string | Physical units |
| `cell_methods` | string | Statistical processing description |
| `valid_min` | number | Minimum valid value |
| `valid_max` | number | Maximum valid value |

Additional supported CF attributes may be included.

The parser preserves these attributes.

Comprehensive CF compliance validation is outside the parser's scope.

### 5.6 Geometry reference

The `geometry` property references the geometry declared in the `geometry` section.

Example:

```yaml
geometry: *projection
```

Constraints:

- Must reference the declared geometry.
- Inline geometry definitions are not permitted.
- Multiple variables may reference the same geometry.

The parser resolves the reference during normalization.

### 5.7 Dimensions

The `dims` property defines the ordered dimensions of the output variable.

Example:

```yaml
dims:
  - *steps
  - *hybrid
  - *y
  - *x
```

Constraints:

- Must be a non-empty sequence.
- Every entry must reference a declared coordinate.
- Duplicate dimensions are not permitted.
- Dimension order must be preserved.
- The dimensions must contain the spatial coordinates required by the variable's geometry.
- Spatial coordinate ordering must be compatible with the geometry.

The output array shape is derived from the referenced coordinate lengths.

For example:

```text
dimensions = [steps, hybrid, y, x]

lengths = [3, 3, 3, 3]

shape = (3, 3, 3, 3)
```

The chunk shape is derived from coordinate chunk sizes:

```text
chunks = (1, 1, 3, 3)
```

### 5.8 Message dimensions

Message dimensions are the variable dimensions excluding those belonging to its spatial geometry.

For example:

```text
Variable dimensions:
    [steps, hybrid, y, x]

Geometry coordinates:
    [y, x]

Message dimensions:
    [steps, hybrid]
```

Each combination of message dimension values identifies a required GRIB2 field.

For the example:

```text
steps = [1, 2, 3]
hybrid = [1, 2, 3]
```

the required combinations are:

```text
(step=1, hybrid=1)
(step=1, hybrid=2)
(step=1, hybrid=3)

(step=2, hybrid=1)
(step=2, hybrid=2)
(step=2, hybrid=3)

(step=3, hybrid=1)
(step=3, hybrid=2)
(step=3, hybrid=3)
```

The number of required GRIB2 messages is the product of the lengths of the message dimensions.

For this example:

```text
3 × 3 = 9 messages
```

This assumes that each GRIB2 message supplies one complete horizontal field.

A variable containing only spatial dimensions requires one GRIB2 message.

---

## 6. References

The configuration uses YAML anchors and aliases to reference coordinates and geometries.

Reference semantics are defined by ADR-0008.

### 6.1 Declaring references

A coordinate definition may declare an anchor:

```yaml
- x: &x
    name: x
    values: [0, 2500, 5000]
    chunk: 3
```

Other definitions may reference it:

```yaml
dims:
  - *x
```

### 6.2 Reference targets

| Property | Required reference type |
|---|---|
| `geometry.*.cf.coords[]` | Coordinate |
| `variables[].geometry` | Geometry |
| `variables[].dims[]` | Coordinate |

References must point to definitions declared in their corresponding top-level sections.

### 6.3 Reference identity

References are validated using the identity of the deserialized YAML mapping.

Two mappings containing identical values are not necessarily references to the same definition.

For example:

```yaml
dims:
  - name: x
    values: [0, 2500, 5000]
    chunk: 3
```

is invalid because it is an inline mapping rather than a reference to a declared coordinate.

### 6.4 Normalized references

After parsing, YAML references are normalized into named or typed references.

For example:

```yaml
geometry: *projection
dims:
  - *steps
  - *hybrid
  - *y
  - *x
```

is conceptually normalized to:

```json
{
  "geometry": "projection",
  "dims": ["steps", "hybrid", "y", "x"]
}
```

YAML anchor names are not part of the normalized configuration model.

---

## 7. Coordinate expressions

Coordinate expressions allow sequences to be generated without explicitly listing every value.

Expression evaluation is performed during configuration parsing.

### 7.1 Supported representations

Explicit values:

```yaml
values: [1, 2, 3, 4, 5]
```

Generated values:

```yaml
values: "range(1, 6)"
```

List comprehension:

```yaml
values: "[x for x in range(1, 6)]"
```

All three produce the same normalized values.

### 7.2 Supported operations

The initial expression language supports:

| Construct | Example |
|---|---|
| Integer literals | `10` |
| Floating-point literals | `2.5` |
| Addition | `x + 1` |
| Subtraction | `x - 1` |
| Multiplication | `x * 2500` |
| Division | `x / 2` |
| Floor division | `x // 2` |
| Exponentiation | `x ** 2` |
| Modulo | `x % 2` |
| Parentheses | `(x + 1) * 2` |
| Range generation | `range(1, 10)` |
| List comprehension | `[x for x in range(10)]` |

The `range()` function supports the standard integer arguments:

```python
range(stop)
range(start, stop)
range(start, stop, step)
```

As in Python, the stop value is excluded.

### 7.3 Examples

Forecast steps:

```yaml
values: "range(1, 67)"
```

Hybrid levels:

```yaml
values: "range(1, 66)"
```

Regular horizontal coordinates:

```yaml
values: "[x * 2500 for x in range(100)]"
```

Coordinates with an offset:

```yaml
values: "[100000 + x * 2500 for x in range(100)]"
```

Fractional coordinates:

```yaml
values: "[x * 0.5 for x in range(20)]"
```

### 7.4 Restricted evaluation

Expressions must be evaluated using a restricted expression evaluator.

Unrestricted Python `eval()` and `exec()` must not be used.

The evaluator must reject:

- Imports.
- Arbitrary function calls.
- Attribute access.
- Subscript access.
- Assignments.
- Lambda expressions.
- Nested comprehensions.
- Generator expressions.
- Access to application objects.
- Filesystem and environment access.

Only explicitly supported syntax and functions may be evaluated.

### 7.5 Resource limits

The evaluator must enforce limits on:

- Expression length.
- Expression nesting depth.
- Number of generated values.
- Intermediate numeric magnitude.
- Computational cost.

Limits must be checked before or during evaluation.

An expression must not be allowed to allocate an excessively large sequence before the size limit is enforced.

### 7.6 Result validation

After evaluation:

- The result must be a finite sequence.
- The sequence must not be empty.
- All values must have supported coordinate types.
- Values must be unique.
- Numeric values must be finite.
- The number of values must not exceed the configured maximum.

The initial implementation supports numeric values generated by expressions.

### 7.7 Normalization

Expressions are evaluated before the normalized configuration model is constructed.

For example:

```yaml
values: "range(1, 4)"
```

becomes:

```python
values = [1, 2, 3]
```

Downstream components must not need to evaluate expressions.

---

## 8. Metadata

The configuration separates GRIB2 and CF metadata.

### 8.1 GRIB2 metadata

GRIB2 metadata is represented by a `grib2` mapping.

Example:

```yaml
grib2:
  discipline: 0
  parameterCategory: 2
  parameterNumber: 1
```

Metadata key names are case-sensitive.

The parser must preserve key names and values.

Supported metadata values include:

- Strings.
- Integers.
- Floating-point numbers.
- Booleans.
- Sequences of supported scalar values.

Nested metadata mappings are not supported unless explicitly defined by the schema.

Null values are not supported in metadata mappings.

### 8.2 CF metadata

CF metadata is represented by a `cf` mapping.

Example:

```yaml
cf:
  standard_name: wind_speed
  units: m s-1
```

Metadata values must preserve their original types.

Additional CF attributes may be supported without changing the overall configuration structure.

### 8.3 Configuration-specific CF structures

Not every property inside a `cf` mapping is an output attribute.

For example:

```yaml
cf:
  crs:
    grid_mapping_name: lambert_conformal_conic
  coords:
    - *y
    - *x
```

The `crs` mapping contains attributes used to construct a CF grid mapping representation.

The `coords` sequence contains references to configuration coordinates.

The converter must interpret these structures rather than copying the complete mapping directly into Zarr attributes.

### 8.4 Metadata validation

The parser performs structural and targeted semantic validation.

This includes:

- Metadata mapping types.
- Required properties.
- Supported scalar types.
- Basic numeric constraints.
- Basic CF coordinate attributes.
- Required CRS properties.

Comprehensive GRIB2 and CF standards validation is not performed by the configuration parser.

Unknown metadata attributes must not be silently discarded.

---

## 9. GRIB2 message selection

The normalized configuration provides the information required to identify GRIB2 messages.

Message selection is derived from:

- Variable GRIB2 metadata.
- Coordinate GRIB2 metadata.
- Coordinate values.
- Geometry GRIB2 metadata.
- Temporal metadata.

The GRIB-reading component is responsible for translating this information into concrete message selection criteria.

### 9.1 Message assignment

Each required combination of non-spatial coordinate values must correspond to exactly one GRIB2 message.

For example:

```text
Variable: wind_speed

Dimensions:
    [steps, hybrid, y, x]

Message dimensions:
    [steps, hybrid]
```

The combination:

```text
step = 2
hybrid = 3
```

must identify exactly one message.

The corresponding field populates:

```python
wind_speed[1, 2, :, :]
```

The output index is determined by the positions of the configured coordinate values.

Message assignment must not depend on the order of messages in the input files.

### 9.2 Missing messages

If no GRIB2 message matches a required combination, conversion must report an error.

The default behavior is not to silently populate missing array slices.

### 9.3 Ambiguous messages

If multiple GRIB2 messages match a required combination, conversion must report an ambiguity error.

The converter must not silently select one of the matches.

### 9.4 Duplicate assignments

A physical GRIB2 message must not populate multiple positions within the same variable.

Overlapping assignments between different variables are rejected by default.

### 9.5 Validation boundary

The parser validates that the configuration can be normalized and used to construct selection criteria.

It does not guarantee that the required messages exist.

Actual message availability, uniqueness, and grid compatibility are validated during GRIB processing.

---

## 10. Normalized configuration model

The parser must produce a normalized representation independent of YAML syntax.

Conceptually:

```text
Configuration
├── coordinates
│   └── Coordinate[]
├── geometry
│   └── Geometry
└── variables
    └── Variable[]
```

### 10.1 Coordinate

```text
Coordinate
├── name: string
├── values: sequence
├── chunk: integer
├── reference_time: optional string
├── grib2: optional mapping
└── cf: optional mapping
```

Derived property:

```python
length = len(values)
```

### 10.2 Geometry

```text
Geometry
├── name: string
├── grib2: optional mapping
├── crs: mapping
└── coordinates: sequence[CoordinateReference]
```

### 10.3 Variable

```text
Variable
├── name: string
├── grib2: mapping
├── cf: optional mapping
├── geometry: GeometryReference
├── dimensions: sequence[CoordinateReference]
├── shape: tuple[int, ...]
├── chunks: tuple[int, ...]
└── message_dimensions: sequence[CoordinateReference]
```

Derived properties:

```python
shape = tuple(
    len(dim.values)
    for dim in dimensions
)

chunks = tuple(
    dim.chunk
    for dim in dimensions
)
```

Message dimensions are derived by excluding the geometry's spatial coordinates from the variable dimensions.

The normalized configuration must not contain unevaluated coordinate expressions or YAML-specific anchor information.

---

## 11. Validation requirements

Validation is performed in multiple stages.

### 11.1 YAML validation

- Valid YAML syntax.
- Safe deserialization.
- Supported anchors and aliases.
- No duplicate mapping keys.
- No unsupported object tags.
- No cyclic structures.

### 11.2 Structural validation

- Required root sections.
- Exactly one geometry definition.
- Correct property types.
- Required properties.
- Valid definition structures.
- Valid metadata mapping structures.
- No unknown structural properties.

### 11.3 Reference validation

- Unique definition names.
- Matching outer keys and names.
- Valid coordinate references.
- Valid geometry references.
- Correct reference categories.
- No duplicate dimension references.
- No duplicate spatial coordinate references.

### 11.4 Expression validation

- Supported syntax.
- Safe evaluation.
- Finite sequences.
- Non-empty results.
- Supported result types.
- Unique values.
- Resource limits.

### 11.5 Targeted semantic validation

- Positive chunk sizes.
- Valid basic CF axis attributes.
- Valid basic CF vertical direction attributes.
- Valid reference-time syntax.
- Required GRIB2 parameter identifiers.
- Basic GRIB2 numeric constraints.
- Geometry-to-variable dimension compatibility.
- Derivable variable shapes and chunk shapes.

### 11.6 Runtime message validation

- Message availability.
- Message selection uniqueness.
- No duplicate assignments.
- Grid compatibility.
- Decoded field shape compatibility.

---

## 12. Error reporting

Validation errors must identify the affected configuration property.

Where possible, errors should contain:

- Configuration path.
- Invalid value.
- Violated constraint.
- YAML line and column.
- Relevant coordinate, geometry, or variable name.

### Structural error

```text
ConfigurationError:
coordinates[1].hybrid.chunk:

Expected a positive integer.
Received: -1
```

### Reference error

```text
ConfigurationError:
variables[0].dims[2]:

Expected a reference to a declared coordinate.
```

### Expression error

```text
ConfigurationError:
coordinates[0].steps.values:

Invalid expression: range(1, 10, 0)

Reason:
range() step must not be zero.
```

### Message selection error

```text
MessageSelectionError:
Variable: wind_speed

Coordinates:
    step = 2
    hybrid = 3

Expected exactly one GRIB2 message.
Found: 2 matching messages.
```

The parser must not silently ignore invalid configuration properties or unsupported expressions.

---

## 13. Complete example

```yaml
coordinates:
  - steps: &steps
      name: steps
      reference_time: "2026-03-27T09:00:00"
      cf:
        axis: T
        standard_name: time
      values: "range(1, 4)"
      chunk: 1

  - hybrid: &hybrid
      name: hybrid
      grib2:
        typeOfFirstFixedSurface: 105
      cf:
        axis: Z
        standard_name: atmosphere_hybrid_sigma_pressure_coordinate
        positive: down
      values: "range(1, 4)"
      chunk: 1

  - x: &x
      name: x
      cf:
        axis: X
        standard_name: projection_x_coordinate
        units: m
      values: "[x * 2500 for x in range(3)]"
      chunk: 3

  - y: &y
      name: y
      cf:
        axis: Y
        standard_name: projection_y_coordinate
        units: m
      values: "[y * 2500 for y in range(3)]"
      chunk: 3

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

### 13.1 Expected normalized coordinates

| Coordinate | Values | Length | Chunk |
|---|---|---:|---:|
| steps | `[1, 2, 3]` | 3 | 1 |
| hybrid | `[1, 2, 3]` | 3 | 1 |
| x | `[0, 2500, 5000]` | 3 | 3 |
| y | `[0, 2500, 5000]` | 3 | 3 |

### 13.2 Expected normalized geometry

```text
name:
    projection

grid_mapping_name:
    lambert_conformal_conic

spatial_dimensions:
    [y, x]
```

### 13.3 Expected normalized variable

```text
name:
    wind_speed

dimensions:
    [steps, hybrid, y, x]

shape:
    (3, 3, 3, 3)

chunks:
    (1, 1, 3, 3)

geometry:
    projection

message_dimensions:
    [steps, hybrid]

required_grib_messages:
    9
```

The expected number of messages assumes one complete horizontal field per GRIB2 message.

---

## 14. Parser acceptance criteria

An implementation conforms to this specification when it:

1. Successfully parses the complete example.
2. Supports YAML anchors and aliases.
3. Validates definition identity and references.
4. Supports explicit coordinate values.
5. Supports restricted coordinate expressions.
6. Produces equivalent normalized values for equivalent explicit and generated sequences.
7. Rejects unsafe or unsupported expressions.
8. Enforces expression resource limits.
9. Preserves GRIB2 and CF metadata.
10. Validates required properties and supported types.
11. Validates variable-to-geometry relationships and rejects more than one geometry definition.
12. Derives variable shapes and chunk shapes.
13. Derives message dimensions.
14. Produces a normalized configuration independent of YAML syntax.
15. Reports invalid configurations with actionable errors.
16. Does not require GRIB input files during parsing.
17. Does not perform GRIB decoding or Zarr writing.

Message selection and assignment validation are separate conversion-stage requirements defined in ADR-0010, sections 8 and 10.

---

## 15. Open specifications

The following behavior requires additional specification before the converter's complete GRIB2 message selection behavior can be considered defined.

### 15.1 Coordinate-to-GRIB2 key mapping

The mapping between coordinate values and GRIB2 keys must be established.

For example:

```yaml
grib2:
  typeOfFirstFixedSurface: 105
values: [1, 2, 3]
```

does not explicitly identify the GRIB2 key against which the values are matched.

The selector implementation must define this relationship.

### 15.2 Forecast time interpretation

The relationship between:

```yaml
reference_time: "2026-03-27T09:00:00"
values: [1, 2, 3]
```

and the GRIB2 forecast-time metadata must be defined.

In particular, the time units and interpretation of the coordinate values must be unambiguous.

### 15.3 GRIB2 metadata combination

The rules for combining variable, coordinate, and geometry GRIB2 metadata into a message selector must be defined.

Potentially conflicting metadata constraints must be handled deterministically.

### 15.4 Message assignment overlap

The current policy rejects assigning the same physical GRIB2 message to multiple output variables.

If intentional sharing of messages becomes necessary, that behavior must be specified explicitly.

### 15.5 Expression resource limits

Concrete limits for expression length, nesting depth, generated sequence length, and computational cost must be documented by the implementation.

### 15.6 Temporal and vertical CF representation

The converter must define how configured forecast times and hybrid coordinates are represented in the output dataset, including any required CF units, auxiliary variables, or formula terms.

These requirements are separate from the structural validity of this configuration format.
