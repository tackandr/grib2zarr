# ADR 0008: Configuration References and Identity

- Status: Proposed  
- Date: 2026-11-11  
- Related: ADR-0007 — Declarative Configuration Architecture for GRIB-to-Zarr Conversion

## Context

The GRIB-to-Zarr converter uses a declarative YAML configuration to describe coordinates, geometries, and variables in the target dataset.

These entities are defined separately to support reuse and avoid duplication.

For example:

- Multiple variables may share the same coordinate dimensions.
- Multiple variables may use the same spatial geometry.
- A geometry references the spatial coordinates defining its grid.
- A variable references its dimensions and associated geometry.

The configuration therefore requires a mechanism for expressing relationships between independently declared definitions.

YAML provides anchors and aliases that allow a mapping to be defined once and referenced elsewhere in the document.

For example:

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

variables:
  - name: example
    dims:
      - *x
```

In this example, `&x` defines a YAML anchor and `*x` references the anchored mapping.

A YAML parser resolves aliases during deserialization. Consequently, an application normally receives references to the corresponding deserialized mappings rather than the original anchor names.

The configuration architecture must establish how these YAML references are interpreted, how definitions are identified, and how references are represented internally.

## Decision

The configuration will use **YAML anchors and aliases to reference coordinate and geometry definitions**.

Definitions will be identified by their declared `name` properties, independently of YAML anchor names.

The parser will resolve YAML aliases, verify that referenced objects correspond to definitions in the appropriate top-level section, and normalize the references into the internal configuration model.

### 1. Definition identity

Coordinates and geometries are declared in their respective top-level sections.

Each definition has a `name` property that identifies it within its category.

For example:

```yaml
coordinates:
  - x: &horizontal_x
      name: x
      values: [0, 2500, 5000]
      chunk: 3
```

The coordinate's identity is `x`, not `horizontal_x`.

The YAML anchor is a serialization mechanism and does not determine the identity of the coordinate.

The following definitions are semantically equivalent:

```yaml
- x: &x
    name: x
    values: [0, 2500, 5000]
    chunk: 3
```

```yaml
- x: &horizontal_coordinate
    name: x
    values: [0, 2500, 5000]
    chunk: 3
```

Changing an anchor name does not change the meaning of the configuration, provided all corresponding aliases are updated.

Coordinate names must be unique within the `coordinates` section.

Geometry names must be unique within the `geometries` section.

Variable names must be unique within the `variables` section.

Names are scoped to their definition category. The same name may therefore occur in different categories.

For coordinates and geometries, the outer mapping key must match the definition's `name` property.

### 2. Coordinate references

Coordinate definitions may be referenced from:

- The `cf.coords` property of a geometry.
- The `dims` property of a variable.

References must use YAML aliases pointing to previously declared coordinate definitions.

Example:

```yaml
coordinates:
  - y: &y
      name: y
      cf:
        axis: Y
        standard_name: projection_y_coordinate
        units: m
      values: [0, 2500, 5000]
      chunk: 3

  - x: &x
      name: x
      cf:
        axis: X
        standard_name: projection_x_coordinate
        units: m
      values: [0, 2500, 5000]
      chunk: 3
```

A geometry can reference these coordinates:

```yaml
geometries:
  - projection: &projection
      name: projection
      cf:
        crs:
          grid_mapping_name: lambert_conformal_conic
        coords:
          - *y
          - *x
```

A variable can reference the same coordinates:

```yaml
variables:
  - name: example
    geometry: *projection
    dims:
      - *y
      - *x
```

The coordinate definitions are shared rather than duplicated.

The order of coordinate references is significant and must be preserved by the parser.

### 3. Geometry references

Variables reference geometry definitions through the `geometry` property.

Example:

```yaml
geometries:
  - projection: &projection
      name: projection
      cf:
        crs:
          grid_mapping_name: lambert_conformal_conic
        coords:
          - *y
          - *x

variables:
  - name: example
    geometry: *projection
    dims:
      - *y
      - *x
```

The `geometry` property must reference a geometry declared in the top-level `geometries` section.

Multiple variables may reference the same geometry.

The geometry definition remains independent of the variables using it.

### 4. References must target declared definitions

A reference is valid only when it points to an object declared in the corresponding top-level collection.

For example, the following is invalid:

```yaml
variables:
  - name: example
    dims:
      - name: x
        values: [0, 2500, 5000]
        chunk: 3
```

Although the dimension entry resembles a coordinate definition, it is not a reference to a declared coordinate.

Similarly, this is invalid:

```yaml
coordinates:
  - x:
      name: x
      values: [0, 2500, 5000]
      chunk: 3

variables:
  - name: example
    dims:
      - name: x
        values: [0, 2500, 5000]
        chunk: 3
```

The inline mapping does not become a valid reference merely because its contents match those of the declared coordinate.

References are based on the identity of the referenced YAML mapping, not on structural equality.

Two mappings with identical contents are not necessarily the same definition.

### 5. Reference resolution

The parser will use a YAML implementation that preserves the identity of aliased mapping objects during deserialization.

After deserialization, the parser will identify the mapping objects corresponding to declared coordinates and geometries.

References will then be validated against these declared objects.

Conceptually, reference resolution proceeds as follows:

1. Deserialize the YAML document.
2. Identify the coordinate definitions in `coordinates`.
3. Identify the geometry definitions in `geometries`.
4. Associate each declared definition's deserialized mapping identity with its name.
5. Inspect references in geometries and variables.
6. Verify that each referenced mapping is one of the declared definitions of the expected category.
7. Replace or normalize references into the corresponding definition names or internal objects.

The parser must not use structural equality to determine whether a reference points to a declared definition.

It must also not rely on the original YAML anchor names being available after deserialization.

### 6. Normalized reference representation

The internal configuration model must not depend on YAML anchors or aliases.

For example, this YAML configuration:

```yaml
variables:
  - name: wind_speed
    geometry: *projection
    dims:
      - *steps
      - *hybrid
      - *y
      - *x
```

will be normalized into a representation conceptually equivalent to:

```json
{
  "name": "wind_speed",
  "geometry": "projection",
  "dims": [
    "steps",
    "hybrid",
    "y",
    "x"
  ]
}
```

Likewise, geometry coordinate references:

```yaml
coords:
  - *y
  - *x
```

will be normalized to:

```json
{
  "coords": ["y", "x"]
}
```

The normalized model may contain either identifiers or resolved typed object references, depending on the implementation.

In either case, YAML-specific anchor names must not be exposed as part of the model's public interface.

### 7. Reference ordering

The order of references in sequences is significant.

For variable dimensions:

```yaml
dims:
  - *steps
  - *hybrid
  - *y
  - *x
```

the corresponding output dimension order is:

```text
(steps, hybrid, y, x)
```

The parser must preserve this order.

Likewise, the order of spatial coordinate references in a geometry must be preserved.

The relationship between a geometry's spatial coordinates and a variable's dimensions is validated according to the configuration schema and parser validation rules.

### 8. Duplicate references

A coordinate must not appear more than once in the `dims` sequence of a single variable.

For example, the following is invalid:

```yaml
dims:
  - *steps
  - *y
  - *y
  - *x
```

Similarly, a geometry must not contain duplicate coordinate references in its `cf.coords` sequence.

However, the same coordinate may be referenced by any number of different variables or geometries.

The same geometry may be referenced by any number of variables.

### 9. Reference scope and declaration order

YAML aliases can refer only to anchors already defined earlier in the document.

Consequently, the configuration must declare referenced definitions before their use.

The expected top-level order is:

```yaml
coordinates:
  # Coordinate definitions

geometries:
  # Geometry definitions referencing coordinates

variables:
  # Variable definitions referencing geometries
  # and coordinates
```

The order of definitions within a section is otherwise not significant, provided YAML alias resolution succeeds.

The normalized configuration model does not impose an additional dependency on declaration order.

### 10. Reference validation errors

Invalid references must produce actionable configuration errors.

Examples include:

- A variable dimension referring to an object that is not a declared coordinate.
- A geometry coordinate referring to an object that is not a declared coordinate.
- A variable geometry referring to an object that is not a declared geometry.
- An inline mapping being used where a reference is required.
- A duplicate coordinate reference in a variable's dimensions.
- A duplicate coordinate reference in a geometry.

For example:

```text
ConfigurationError:
variables[0].dims[2]:
Expected a reference to a declared coordinate.
```

Or:

```text
ConfigurationError:
variables[0].geometry:
Expected a reference to a geometry declared
in the geometries section.
```

Undefined YAML aliases are rejected during YAML deserialization.

The parser should preserve the distinction between YAML syntax errors and configuration validation errors.

The detailed error-reporting strategy is defined in ADR-0004.

## Alternatives considered

### 1. Name-based references

References could use coordinate and geometry names directly:

```yaml
variables:
  - name: wind_speed
    geometry: projection
    dims:
      - steps
      - hybrid
      - y
      - x
```

**Advantages:**

- References are simple strings.
- The parser does not need to preserve YAML object identity.
- Configurations are easier to generate programmatically.
- References do not depend on declaration order.
- The format can be represented directly in JSON.

**Disadvantages:**

- References require explicit name lookup.
- Misspelled names cannot be detected by YAML deserialization.
- The configuration loses YAML's native alias mechanism.

**Not selected:** The existing configuration uses YAML anchors and aliases, which already provide reusable references without introducing an additional reference syntax.

### 2. Supporting both YAML aliases and name-based references

The parser could accept both forms:

```yaml
dims:
  - *x
  - y
```

**Advantages:**

- Greater flexibility for configuration authors.
- Easier migration between reference mechanisms.
- Better support for programmatically generated configurations.

**Disadvantages:**

- Multiple representations for the same relationship.
- More complex validation and normalization.
- Additional parser test cases.
- Potential inconsistencies in configuration style.

**Not selected:** A single reference mechanism keeps the configuration format and parser behavior simpler.

Support for name-based references may be reconsidered if programmatic configuration generation becomes a significant requirement.

### 3. Embedding definitions directly in variables

Each variable could contain its complete coordinate and geometry definitions.

**Advantages:**

- No reference-resolution mechanism required.
- Variable definitions would be self-contained.

**Disadvantages:**

- Shared definitions would be duplicated.
- Different variables could define inconsistent versions of the same coordinate.
- Configuration files would become larger.
- Modifying a shared geometry would require multiple changes.

**Rejected:** This conflicts with the reusable-definition architecture established in ADR-0001.

### 4. Using YAML anchor names as definition identifiers

The parser could treat anchor names such as `&x` and `&projection` as the identifiers of their definitions.

**Advantages:**

- Avoids requiring a separate `name` property for identification.
- Makes references and identifiers visually identical.

**Disadvantages:**

- YAML libraries do not necessarily expose anchor names after deserialization.
- The internal model would depend on YAML serialization details.
- Renaming an anchor would change the definition's identity.
- The configuration could not be normalized independently of YAML syntax.

**Rejected:** Definition identity must be independent of the YAML reference mechanism.

## Consequences

### Positive

- Coordinates and geometries can be reused across multiple variables.
- Shared definitions have consistent identity.
- YAML aliases express relationships without duplicating configuration content.
- Reference validation can distinguish declared definitions from arbitrary inline mappings.
- The normalized configuration model is independent of YAML anchors.
- Reference ordering is preserved for multidimensional arrays.
- The parser has one unambiguous reference mechanism.

### Negative

- The parser must preserve mapping object identity during reference resolution.
- YAML aliases require referenced definitions to appear earlier in the document.
- The configuration is more tightly coupled to YAML than a name-based representation would be.
- Programmatically generated configurations must produce valid YAML anchors and aliases.
- Authors must understand YAML anchor and alias syntax.
- Supporting alternative serialization formats would require a different external reference representation.

## Scope

This ADR defines:

- Definition identity.
- YAML alias-based references.
- Valid reference targets.
- Reference resolution.
- Reference normalization.
- Reference ordering.
- Duplicate reference constraints.

It does not define:

- The complete coordinate, geometry, or variable schemas.
- The detailed semantics of GRIB2 or CF metadata.
- The rules for determining whether a variable's dimensions are compatible with its geometry.
- The internal implementation language or data model classes.
- The complete configuration validation pipeline.
- GRIB-to-Zarr conversion behavior.

These concerns are addressed by the configuration schema and subsequent ADRs.

## Related documents

- **ADR-0007:** Declarative Configuration Architecture for GRIB-to-Zarr Conversion
- **ADR-0009:** GRIB2 and CF Metadata Model *(planned)*
- **ADR-0010:** Configuration Parsing and Validation *(planned)*
- **Configuration Schema:** Detailed YAML structure and field definitions *(planned)*
