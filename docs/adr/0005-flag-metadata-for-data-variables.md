# ADR 0005: Flag metadata for data variables

- Status: Accepted
- Date: 2026-10-19
- Deciders: Andreas Tack
- Extends: [ADR 0004: Datasets data variable metadata](0004-datasets-data-variable-metadata.md)

## Context

[ADR 0004](0004-datasets-data-variable-metadata.md) establishes requirements for identifying the meaning and units of data variables using CF Conventions metadata.

Data variables may also contain flag values, such as quality control indicators, status codes, or bitwise condition flags. The meaning of these values cannot be determined reliably from the variable name or numeric values alone. ADR 0004 does not explicitly require metadata describing individual flags.

## Decision

In addition to the requirements established in [ADR 0004](0004-datasets-data-variable-metadata.md), data variables containing flag values MUST provide metadata defining their meanings using the flag attributes specified by the CF Conventions.

This decision extends ADR 0004 without replacing or changing any of its requirements.

### Rules

Valid source data MUST satisfy the following additional requirements:

- A data variable containing flag values MUST provide a CF `flag_meanings` attribute and at least one of the CF `flag_values` or `flag_masks` attributes.
- For enumerated flags, `flag_values` MUST specify the encoded values, and `flag_meanings` MUST specify a corresponding meaning for each value.
- For bitwise flags, `flag_masks` MUST specify the masks used to identify the flags, and `flag_meanings` MUST specify a corresponding meaning for each mask.
- A data variable MAY provide both `flag_values` and `flag_masks` where the combined representation is appropriate under the CF Conventions.
- The flag attributes MUST follow the CF Conventions requirements for their types, structure, and correspondence between values or masks and meanings.

These requirements ensure that flag values can be interpreted consistently without relying on external documentation or source-specific conventions.

## Consequences

Source data containing flag values must provide sufficient metadata to interpret the defined flags.

Data that contains flag values but lacks the required flag metadata must be normalized before it can be processed as valid source data.

Both enumerated and bitwise flags can be represented using CF-compatible metadata. All existing requirements of ADR 0004, including explicit `units`, continue to apply.
