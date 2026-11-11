# ADR 0002: Time dimension of datasets

- Status: Accepted
- Date: 2026-09-07
- Deciders: Andreas Tack

## Context

Temporal coordinates are used when processing, combining, and publishing data. To reliably identify and interpret time dimensions in processed data, temporal coordinates must be described using standardized metadata.

Forecast data introduces an additional distinction between forecast reference time, forecast lead time, and forecast valid time. A consistent representation is required so that these concepts can be identified without relying on naming conventions or other heuristics.

## Decision

We use the CF Conventions to describe the time axes of our data, with additional restrictions to remove ambiguities and provide a consistent representation of temporal coordinates.

### Rules

Valid temporal source data MUST satisfy the following requirements:

- Each data variable MUST contain exactly one time dimension.
- Each time dimension MUST have an associated one-dimensional coordinate variable that uses the corresponding time dimension as its only dimension and has the attribute `axis="T"`.
- The time coordinate MUST represent valid time and describe it according to the CF Conventions, including CF-compliant time units containing a reference date/time. A `calendar` attribute SHOULD be provided where applicable.
- A `Dataset` MAY contain multiple time dimensions, and different data variables MAY use different time dimensions.
- A time dimension MAY have size `1`. Single-time data MUST be represented using such a size-one time dimension rather than a scalar valid-time coordinate.
- If the `Dataset` represents forecast data, it MUST contain exactly one scalar coordinate with the CF standard name `forecast_reference_time`.
- The `forecast_reference_time` coordinate MUST describe the forecast analysis or initialization time using CF-compliant time metadata.
- For forecast data, the values of the `axis="T"` coordinate MUST represent forecast valid times, not forecast lead times. Forecast lead time MAY be provided as an additional coordinate or auxiliary coordinate but MUST NOT replace the valid-time coordinate.

These requirements are stricter than the CF Conventions. In particular, CF does not require the `axis` attribute and permits time coordinates to be identified through other metadata. The additional constraints provide an unambiguous mechanism for identifying the valid-time dimension of each data variable.

For forecast datasets, the scalar `forecast_reference_time` represents the analysis or initialization time of the forecast, while the `axis="T"` coordinate represents the times at which the forecast values are valid.

## Consequences

Source data must use a data model and format capable of preserving coordinate variables and their associated attributes.

Data that does not provide the required CF metadata, or represents temporal information using a different structure, must be normalized before it can be processed as valid source data.

The stricter representation reduces flexibility compared with the CF Conventions but allows time dimensions and forecast temporal semantics to be identified deterministically without relying on variable names or heuristics.
