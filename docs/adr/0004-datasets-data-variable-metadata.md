# ADR 0004: Datasets data variable metadata

- Status: Accepted
- Date: 2026-09-07
- Deciders: Andreas Tack

## Context

Reading, processing, combining, and publishing data requires a consistent way to identify the physical quantity represented by each data variable. Variable names alone are not sufficient because they may vary between data sources and processing systems.

Standardized metadata is therefore required to describe the meaning and units of each data variable independently of its variable name.

## Decision

We use CF Conventions metadata to describe the parameters represented by data variables, with additional requirements to ensure that their meaning and units can be identified deterministically.

### Rules

Valid source data MUST satisfy the following requirements:

- Each data variable MUST contain at least one of the following attributes describing the represented parameter: a CF `standard_name` or a `long_name`.
- A CF `standard_name` SHOULD be used whenever an applicable CF standard name exists. A `long_name` MAY be used when no suitable CF standard name is available.
- A data variable MAY contain both `standard_name` and `long_name`. In this case, `standard_name` defines the standardized meaning of the parameter and `long_name` MAY provide a more descriptive human-readable label.
- Each data variable MUST contain a `units` attribute describing the units of its values.
- The `units` attribute MUST use units compatible with the CF Conventions.
- If the parameter is dimensionless, the data variable MUST explicitly declare `units="1"` rather than omitting the `units` attribute.
- If a `standard_name` is provided, the declared units MUST be dimensionally compatible with the canonical units defined for that CF standard name.

These requirements are stricter than the CF Conventions. In particular, requiring every data variable to provide explicit units ensures that the absence of a `units` attribute cannot be confused with a dimensionless quantity. Using `units="1"` for dimensionless parameters provides an explicit and machine-readable representation.

The use of CF `standard_name` where available also provides a standardized semantic identifier for parameters, while `long_name` allows parameters not covered by the CF Standard Name Table to be represented without relying on variable names.

## Consequences

Source data must provide sufficient metadata to identify both the meaning and units of every data variable.

Data that lacks parameter metadata or explicit units must be normalized before it can be processed as valid source data.

Parameters that do not have an applicable CF `standard_name` can still be represented using `long_name`, but such parameters have weaker semantic interoperability because their meaning is expressed as free text.
