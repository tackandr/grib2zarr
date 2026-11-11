# ADR 0010: Configuration Parsing and Validation

-Status: Proposed  
-Date: 2026-11-11  
-Related: ADR-0007, ADR-0008, ADR-0009

## Context

The GRIB-to-Zarr converter uses a declarative YAML configuration describing the coordinates, geometries, and variables of the target Zarr dataset.

The configuration defines the structure of the output dataset and provides GRIB2 metadata used to identify the messages containing its data.

GRIB2 messages typically contain two-dimensional fields associated with a particular parameter, forecast time, vertical level, and horizontal grid.

The converter assembles these fields into multidimensional Zarr arrays.

For example, a variable with dimensions:

```text
(steps, hybrid, y, x)
```

requires one GRIB2 message for every combination of forecast step and hybrid level, assuming each message contains the complete horizontal field.

The configuration parser must produce a representation from which these message selections and output array positions can be determined.

The configuration must also support concise coordinate definitions. Explicitly listing coordinate values is practical for short sequences but inconvenient for large or regularly spaced coordinates.

For example:

```yaml
values: [1, 2, 3, 4, 5]
```

could instead be expressed as:

```yaml
values: "range(1, 6)"
```

or:

```yaml
values: "[x for x in range(1, 6)]"
```

The parser therefore needs to support both explicit coordinate values and generated coordinate values.

Validation must ensure that the configuration is structurally correct and sufficiently unambiguous for the converter to select GRIB2 messages.

However, message availability and uniqueness cannot be fully verified without examining the actual GRIB input.

The architecture must distinguish configuration validation from GRIB message validation.

## Decision

The converter will implement configuration parsing and validation as a separate processing stage before GRIB message selection and Zarr writing.

The parser will:

1. Deserialize the YAML configuration.
2. Validate its structure.
3. Register coordinate and geometry definitions.
4. Resolve references between definitions.
5. Evaluate coordinate value expressions.
6. Perform targeted semantic validation.
7. Construct a normalized configuration model.
8. Derive output dimensions, shapes, chunk sizes, and message dimensions.

The normalized configuration model will contain concrete coordinate values and resolved references.

The parser will not require access to GRIB input files.

The GRIB-reading component will subsequently use the normalized configuration to select messages and validate their assignments to output arrays.

Validation will be limited to structural and targeted semantic checks. Comprehensive GRIB2 and CF standards validation is outside the parser's scope.

## 1. Parsing architecture

Configuration processing will follow this pipeline:

```text
               YAML configuration
                       |
                       v
               YAML deserialization
                       |
                       v
               Structural validation
                       |
                       v
              Definition registration
                       |
                       v
               Reference resolution
                       |
                       v
           Coordinate expression evaluation
                       |
                       v
             Targeted semantic validation
                       |
                       v
            Normalized configuration model
                       |
                       v
                GRIB message index
                       |
                       v
                Message selection
                       |
                       v
             Message assignment validation
                       |
                       v
                 GRIB decoding
                       |
                       v
                  Zarr writing
```

The parser is responsible for all stages through construction of the normalized configuration model.

GRIB message indexing, selection, assignment validation, decoding, and Zarr writing belong to the conversion process.

## 2. YAML deserialization

The configuration will be deserialized using a YAML parser supporting anchors and aliases.

Deserialization must:

- Use safe YAML loading.
- Preserve mapping and sequence structure.
- Preserve the identity of aliased mappings.
- Reject malformed YAML.
- Reject duplicate mapping keys.
- Reject unsupported YAML tags.
- Reject cyclic structures.

YAML aliases must resolve to declared definitions as established in ADR-0002.

The parser must not depend on retaining YAML anchor names after deserialization.

## 3. Structural validation

Structural validation verifies that the configuration follows the expected schema.

### 3.1 Root structure

The root document must contain three sections:

```yaml
coordinates: []
geometries: []
variables: []
```

Each section must be a sequence.

Unknown top-level sections are rejected.

At least one variable must be defined.

### 3.2 Coordinate definitions

Each coordinate must contain:

- A unique name.
- A `values` property.
- A positive integer `chunk` property.

Coordinates may additionally contain:

- GRIB2 metadata.
- CF metadata.
- Reference-time information.

The parser must verify that:

- Coordinate names are unique.
- The outer definition key matches the `name` property.
- The `values` property is either a YAML sequence or an expression string.
- Chunk sizes are positive integers.
- Optional metadata mappings have the expected structure.

### 3.3 Geometry definitions

Each geometry must contain:

- A unique name.
- A CF coordinate reference system definition.
- References to its spatial coordinates.

The parser must verify that:

- Geometry names are unique.
- The outer definition key matches the `name` property.
- The CRS definition is a mapping.
- The CRS contains a `grid_mapping_name`.
- Spatial coordinate references are valid.
- Referenced coordinates are declared in the `coordinates` section.
- Spatial coordinate references are not duplicated.

### 3.4 Variable definitions

Each variable must contain:

- A unique name.
- GRIB2 parameter identification metadata.
- A geometry reference.
- An ordered sequence of dimension references.

The parser must verify that:

- Variable names are unique.
- Required GRIB2 properties are present.
- The geometry reference resolves.
- All dimension references resolve.
- Dimensions are not duplicated.
- The dimension sequence is non-empty.

Variable dimension order must be preserved.

## 4. Reference validation

References are resolved according to ADR-0008.

The parser must verify that:

- Geometry coordinate references identify declared coordinates.
- Variable dimension references identify declared coordinates.
- Variable geometry references identify declared geometries.
- References point to the correct definition category.
- Inline mappings are not accepted as substitutes for declared references.

References must be validated using the identity of deserialized YAML mappings rather than structural equality.

After validation, references are normalized into names or typed object references.

The normalized configuration model must not depend on YAML anchors.

## 5. Coordinate value expressions

The `values` property may contain either an explicit YAML sequence or a string expression that generates a sequence.

### 5.1 Explicit values

Example:

```yaml
coordinates:
  - steps: &steps
      name: steps
      values: [1, 2, 3, 4, 5]
      chunk: 1
```

Explicit values are preserved in their declared order.

### 5.2 Generated values

Example:

```yaml
coordinates:
  - steps: &steps
      name: steps
      values: "[x for x in range(1, 6)]"
      chunk: 1
```

The parser evaluates the expression and generates:

```python
[1, 2, 3, 4, 5]
```

Both representations must result in the same normalized coordinate model.

### 5.3 Supported expression language

Expressions will use a restricted subset of Python expression syntax.

The initial implementation will support:

- Integer and floating-point literals.
- Arithmetic operators: `+`, `-`, `*`, `/`, `//`, `**`, and `%`.
- Parentheses for grouping.
- The `range()` function.
- List comprehensions with a single iteration variable.
- Arithmetic expressions within list comprehensions.

Examples:

```yaml
# Integer sequence
values: "range(10)"

# Explicit start and stop
values: "range(1, 66)"

# Explicit step
values: "range(0, 100, 5)"

# List comprehension
values: "[x for x in range(1, 66)]"

# Arithmetic transformation
values: "[x * 0.5 for x in range(1, 21)]"

# Regularly spaced horizontal coordinates
values: "[x * 2500 for x in range(100)]"
```

Expressions must produce finite sequences.

The following constructs are not supported:

- Arbitrary function calls.
- Imports.
- Attribute access.
- Subscript access.
- Assignments.
- Lambda expressions.
- Nested comprehensions.
- Generator expressions.
- Access to external resources.
- Access to Python built-ins other than explicitly supported functions.

The expression language is intentionally limited to deterministic coordinate generation.

### 5.4 Safe expression evaluation

Expressions must not be evaluated using unrestricted Python `eval()` or `exec()`.

The implementation must use a restricted expression evaluator.

An implementation may use Python's abstract syntax tree (AST) to recognize and evaluate only explicitly supported operations.

The evaluator must not expose arbitrary Python objects, modules, functions, or built-ins.

Expression evaluation must be deterministic and independent of external state.

The evaluator must enforce resource limits, including:

- Maximum expression length.
- Maximum expression nesting depth.
- Maximum number of generated coordinate values.
- Maximum size of intermediate numeric values.
- Maximum arithmetic operation count or equivalent computational budget.

These limits must be enforced before or during evaluation, rather than only after a large result has been materialized.

The exact limits are implementation-defined and must be documented and tested.

### 5.5 Expression result validation

After evaluation, generated values must satisfy the same validation requirements as explicitly provided coordinate values.

The parser must verify that:

1. The result is a finite sequence.
2. The sequence is not empty.
3. Every value has a supported coordinate type.
4. Coordinate values are unique.
5. Numeric values are finite.
6. The number of values does not exceed the configured limit.

Boolean values must not be accepted as numeric coordinates.

The initial implementation supports numeric coordinate values generated by expressions.

### 5.6 Normalization

Expressions are evaluated during parsing.

For example:

```yaml
values: "range(1, 4)"
```

is normalized to:

```python
values = [1, 2, 3]
```

Downstream components must not need to evaluate expressions.

Coordinate lengths, variable shapes, chunk shapes, and GRIB2 message selection combinations are always derived from the normalized values.

### 5.7 Expression evaluation scope

Expression evaluation applies only to coordinate `values`.

Other strings must remain literal.

For example:

```yaml
cf:
  standard_name: wind_speed
  units: m s-1
```

must not be interpreted as expressions.

GRIB2 metadata, geometry definitions, and variable references must not be evaluated as expressions.

## 6. Targeted semantic validation

Targeted semantic validation checks constraints directly relevant to constructing output arrays and selecting GRIB messages.

The parser will not implement comprehensive GRIB2 or CF standards validation.

### 6.1 Coordinates

The parser must verify that:

- Normalized coordinate values are non-empty.
- Coordinate values are unique.
- Numeric coordinate values are finite.
- Coordinate values have supported types.
- Chunk sizes are positive integers.
- CF `axis`, when present, is one of `T`, `Z`, `Y`, or `X`.
- CF `positive`, when present, is `up` or `down`.
- Reference times, when present, follow the supported date-time format.

The parser must preserve coordinate value order.

Coordinate values need not be monotonically increasing unless required by a specific coordinate type or downstream operation.

### 6.2 Geometries

The parser must verify that:

- A geometry defines spatial coordinates.
- Spatial coordinates are valid references.
- Geometry coordinates appear in the dimensions of variables using that geometry.
- The relative order of spatial coordinates matches the geometry definition.

For example:

```yaml
cf:
  coords:
    - *y
    - *x
```

requires variables using that geometry to contain both `y` and `x` in the same relative order.

The parser does not perform projection calculations or verify grid compatibility against GRIB input.

### 6.3 Variables

The parser must verify that:

- Required GRIB2 parameter identification fields are present.
- GRIB2 parameter identifiers have valid structural types and basic numeric constraints.
- Referenced dimensions are unique.
- Geometry coordinates are included in the variable dimensions.
- Variable shape can be derived from coordinate definitions.
- Variable chunk shape can be derived from coordinate definitions.

For example:

```text
dims   = [steps, hybrid, y, x]
shape  = [3, 3, 3, 3]
chunks = [1, 1, 3, 3]
```

The parser must not require GRIB input files to validate these properties.

## 7. Normalized configuration model

The parser will produce a normalized configuration model independent of YAML-specific syntax.

Conceptually:

```text
Configuration
├── coordinates: Coordinate[]
├── geometries: Geometry[]
└── variables: Variable[]
```

### 7.1 Coordinate

```text
Coordinate
├── name
├── values
├── chunk
├── reference_time
├── grib2
└── cf
```

Derived property:

```python
length = len(values)
```

All coordinate values must be concrete values.

The normalized model must not contain unevaluated coordinate expressions.

### 7.2 Geometry

```text
Geometry
├── name
├── grib2
├── crs
└── coordinates
```

Coordinate references must be resolved.

### 7.3 Variable

```text
Variable
├── name
├── grib2
├── cf
├── geometry
├── dimensions
├── shape
├── chunks
└── message_dimensions
```

The parser derives:

```python
shape = tuple(len(dim.values) for dim in dimensions)

chunks = tuple(dim.chunk for dim in dimensions)
```

Message dimensions are the variable dimensions excluding those associated with the horizontal geometry.

For example:

```text
dimensions         = [steps, hybrid, y, x]
message_dimensions = [steps, hybrid]
```

The normalized model must retain the information required to construct GRIB2 message selection criteria.

## 8. GRIB2 message selection model

The converter uses the normalized configuration to determine which GRIB2 messages populate each variable.

Message selection is derived from:

- Variable GRIB2 metadata.
- Coordinate GRIB2 metadata.
- Coordinate values.
- Geometry GRIB2 metadata.
- Applicable temporal metadata.

Conceptually:

```text
Variable metadata
        +
Coordinate metadata and values
        +
Geometry metadata
        +
Temporal information
        |
        v
GRIB2 message selection criteria
```

The precise mapping of configuration properties to GRIB2 keys belongs to the GRIB-reading component.

### 8.1 Coordinate combinations

For a variable with dimensions:

```text
(steps, hybrid, y, x)
```

the `y` and `x` dimensions describe the horizontal field.

The remaining dimensions identify the combinations of GRIB2 messages required.

For example:

```text
steps  = [1, 2, 3]
hybrid = [1, 2, 3]
```

The required combinations are:

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

The variable requires nine GRIB2 messages, assuming each message supplies one complete horizontal field.

Its output array shape is:

```text
(3, 3, 3, 3)
```

### 8.2 Generalized message dimensions

The converter must not assume that every variable has exactly four dimensions.

For a variable \(v\), let:

- \(D_v\) represent its dimensions.
- \(S_v\) represent the dimensions associated with its spatial geometry.
- \(M_v\) represent its remaining dimensions.

Then:

\[
M_v = D_v \setminus S_v
\]

The expected number of GRIB2 messages is:

\[
N_v = \prod_{d \in M_v}|d|
\]

where \(|d|\) is the number of coordinate values.

This assumes that each GRIB2 message supplies one complete horizontal field.

A variable containing only spatial dimensions requires one GRIB2 message.

The converter is responsible for translating the non-spatial coordinate combinations into message selection criteria.

## 9. GRIB2 message uniqueness

Each configured variable must identify an unambiguous set of GRIB2 messages.

For every combination of non-spatial coordinate values, exactly one GRIB2 message must be selected.

A GRIB2 message must not populate multiple positions within the same variable.

This establishes a one-to-one correspondence between required coordinate combinations and selected GRIB2 messages.

### 9.1 Missing messages

If no message matches a required coordinate combination, the converter must report a missing-message error.

Example:

```text
Variable: wind_speed

Coordinates:
    step = 2
    hybrid = 3

Matching messages: 0
```

The default behavior is to reject incomplete input rather than silently fill the corresponding output slice.

Alternative missing-data policies require a separate decision.

### 9.2 Ambiguous messages

If more than one message matches the same coordinate combination, the converter must report an ambiguity error.

Example:

```text
Variable: wind_speed

Coordinates:
    step = 2
    hybrid = 3

Matching messages: 2
```

The converter must not silently select the first or last matching message.

### 9.3 Duplicate assignments

A physical GRIB2 message must not be assigned to multiple positions within the same variable.

Overlapping assignments between different variables are also rejected by default.

This establishes distinct message ownership for each variable.

Intentional sharing of source messages between variables may be introduced through a separate architectural decision.

### 9.4 Validation timing

Message uniqueness cannot generally be established during configuration parsing.

It depends on the metadata and availability of actual GRIB messages.

The configuration parser may detect duplicate or contradictory selection definitions that are evident from the configuration alone.

Complete message uniqueness validation occurs after GRIB message metadata has been indexed.

## 10. GRIB message assignment validation

The GRIB-reading component will construct a message assignment plan before decoding field data whenever feasible.

The assignment plan maps each required non-spatial coordinate combination to a physical GRIB2 message.

For example:

```text
Variable: wind_speed

Output position          GRIB message
-------------------------------------
[0, 0, :, :]             Message 101
[0, 1, :, :]             Message 102
[0, 2, :, :]             Message 103

[1, 0, :, :]             Message 104
[1, 1, :, :]             Message 105
[1, 2, :, :]             Message 106

[2, 0, :, :]             Message 107
[2, 1, :, :]             Message 108
[2, 2, :, :]             Message 109
```

The assignment plan must satisfy the following constraints:

1. Every required coordinate combination has exactly one message.
2. No message is assigned to multiple positions within one variable.
3. Different variables have non-overlapping message assignments.
4. Selected messages match the expected geometry.
5. Decoded fields are compatible with the configured horizontal dimensions.

Message selection must not depend on the order in which messages occur in the input files.

If multiple physical messages match the same criteria, the assignment is ambiguous even when their data values are identical.

## 11. Error handling

Errors must identify the affected configuration property or GRIB message assignment.

### 11.1 Structural errors

Example:

```text
ConfigurationError:
variables[0].dims[2]:

Expected a reference to a declared coordinate.
```

### 11.2 Expression errors

Example:

```text
ConfigurationError:
coordinates[0].steps.values:

Invalid expression: range(1, 10, 0)

Reason:
range() step must not be zero.
```

Unsupported expressions must be rejected without executing prohibited operations.

### 11.3 Semantic errors

Example:

```text
ConfigurationError:
coordinates[1].hybrid.values:

Duplicate coordinate value: 3
```

### 11.4 Message selection errors

Example:

```text
MessageSelectionError:
Variable: wind_speed

Coordinates:
    step = 2
    hybrid = 3

Expected exactly one GRIB2 message.
Found: 2 matching messages.
```

### 11.5 Error reporting requirements

Errors should include:

- The affected configuration path or variable.
- The invalid value or coordinate combination.
- A description of the violated constraint.
- YAML line and column numbers when available.
- GRIB file identifiers and message offsets for message assignment errors when available.

The parser must not silently ignore malformed configuration properties or invalid expressions.

## 12. Validation scope

Validation is divided between the parser and the conversion components.

### Configuration parser

The parser performs:

- YAML syntax validation.
- Structural schema validation.
- Definition name validation.
- Reference resolution.
- Coordinate expression evaluation.
- Coordinate value validation.
- Chunk size validation.
- Basic GRIB2 metadata constraints.
- Basic CF metadata constraints.
- Geometry and dimension relationship validation.
- Shape and chunk derivation.

### GRIB-reading component

The GRIB-reading component performs:

- GRIB2 metadata interpretation.
- Message indexing and selection.
- Missing-message detection.
- Ambiguous-message detection.
- Message assignment uniqueness validation.
- Grid consistency validation.
- Field shape compatibility validation.

### Output validation

Output validation may perform:

- CF standard-name checks.
- Unit compatibility checks.
- Grid mapping validation.
- Vertical coordinate formula validation.
- Dataset-level CF compliance checking.

Comprehensive GRIB2 and CF standards validation is not required during configuration parsing.

## 13. Alternatives considered

### 13.1 Perform all validation during parsing

The parser could validate GRIB2 message availability, metadata semantics, and output compliance.

**Rejected because:**

- Message availability depends on actual input files.
- Parsing would become coupled to GRIB decoding.
- Comprehensive standards validation introduces substantial complexity.
- Configuration validity should be independently verifiable.

### 13.2 Require explicit coordinate values

Coordinates could be restricted to YAML sequences.

**Advantages:**

- Simpler parser.
- No expression evaluation.
- Straightforward validation.

**Disadvantages:**

- Large regular coordinate sequences are verbose.
- Configuration maintenance becomes cumbersome.
- Repetitive values must be generated externally.

**Not selected:** Restricted expressions offer concise, deterministic coordinate generation.

### 13.3 Use unrestricted Python evaluation

Coordinate expressions could be evaluated directly using Python's `eval()`.

**Rejected because:**

- It permits arbitrary code execution.
- Expressions could access external state.
- Configuration behavior would be difficult to constrain.
- Untrusted configuration files would create security risks.

A restricted evaluator is required instead.

### 13.4 Use structured coordinate generators

Coordinate values could be generated using declarative mappings:

```yaml
values:
  range:
    start: 1
    stop: 66
    step: 1
```

**Advantages:**

- Straightforward schema validation.
- No expression interpreter.
- Language-independent representation.

**Disadvantages:**

- More verbose.
- Arithmetic transformations require additional schema constructs.
- Every new generation pattern requires extending the format.

**Not selected:** Restricted expressions provide greater flexibility while retaining deterministic behavior.

### 13.5 Infer dimensions from GRIB messages

The converter could derive output dimensions from the input data.

**Rejected because:**

- Output structure would depend on available GRIB messages.
- Missing fields could alter the resulting dimensions.
- Dimension order would require inference.
- Reproducibility would be reduced.

Configured coordinate values remain authoritative.

### 13.6 Allow ambiguous message selection

The converter could select the first matching message when multiple messages satisfy the same criteria.

**Rejected because:**

- Results could depend on input ordering.
- Incorrect fields could be silently selected.
- Reproducibility would be reduced.

Every required coordinate combination must identify exactly one message.

### 13.7 Allow missing messages by default

The converter could fill unmatched array slices with missing values.

**Not selected:** Missing input data must be reported rather than silently accepted.

A configurable missing-data policy may be introduced separately.

## Consequences

### Positive

- Configuration validity can be checked without GRIB input files.
- Coordinate sequences can be represented concisely.
- Explicit and generated coordinate values have identical downstream semantics.
- Expressions are evaluated before GRIB processing.
- Output dimensions and chunk sizes are deterministic.
- The parser remains independent of GRIB decoding and Zarr writing.
- Every required output field has an unambiguous relationship to a GRIB2 message.
- Missing and ambiguous message selections are detected.
- Input message ordering does not affect the output array structure.
- Validation responsibilities are clearly separated.

### Negative

- The parser requires a restricted expression evaluator.
- Expression evaluation introduces additional security and resource-management requirements.
- The supported expression language must be documented and maintained.
- Message uniqueness cannot be guaranteed during configuration parsing alone.
- A separate message assignment validation stage is required.
- Strict missing-message and uniqueness requirements may reject otherwise usable GRIB inputs.
- Some metadata consistency errors may only be detected during conversion.

## Scope

This ADR defines:

- Configuration parsing stages.
- Structural validation.
- Reference validation.
- Restricted coordinate expression evaluation.
- Targeted semantic validation.
- Normalized configuration construction.
- Derivation of dimensions, shapes, and chunks.
- GRIB2 message selection requirements.
- Message uniqueness and assignment validation.
- Error handling responsibilities.

This ADR does not define:

- The complete configuration schema.
- Comprehensive GRIB2 code-table validation.
- Comprehensive CF compliance validation.
- The exact mapping of coordinate values to GRIB2 keys.
- GRIB decoding algorithms.
- Projection transformations.
- Vertical coordinate transformations.
- Zarr serialization details.
- The implementation language or specific parser libraries.

## Related documents

- **ADR-0007:** Declarative Configuration Architecture for GRIB-to-Zarr Conversion
- **ADR-0008:** Configuration References and Identity
- **ADR-0009:** GRIB2 and CF Metadata Model
- **Configuration Schema:** Detailed YAML structure and field definitions
