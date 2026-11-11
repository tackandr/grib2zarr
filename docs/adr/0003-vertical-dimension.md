# ADR 0003: Vertical dimension of datasets

- Status: Accepted
- Date: 2026-09-07
- Deciders: Andreas Tack

## Context

Reading, processing, combining, and publishing data requires a well-defined representation of the vertical reference system. To reliably identify and interpret vertical coordinates in processed data, vertical dimensions and their reference information must be described using standardized metadata.

Vertical coordinates may represent different physical or model-based quantities, such as height, depth, pressure, or parameterized model levels. A consistent representation is required so that the processing system can identify the vertical dimension and interpret its reference without relying on variable names or other heuristics.

## Decision

We use the CF Conventions to describe the vertical coordinates of our data, with additional restrictions to remove ambiguities and provide a consistent representation of vertical dimensions and their reference systems.

### Rules

Valid vertical source data MUST satisfy the following requirements:

- Each data variable MUST contain exactly one vertical dimension.
- Each vertical dimension MUST have an associated one-dimensional coordinate variable that uses the corresponding vertical dimension as its only dimension and has the attribute `axis="Z"`.
- A `Dataset` MAY contain multiple vertical dimensions, and different data variables MAY use different vertical dimensions.
- A vertical dimension MAY have size `1`. Single-level data, such as surface data, MUST be represented using such a size-one vertical dimension rather than a scalar vertical coordinate.
- The vertical coordinate and any supporting variables MUST provide sufficient metadata to describe the vertical reference according to the CF Conventions.

These requirements are stricter than the CF Conventions. In particular, CF does not require the `axis` attribute and permits vertical coordinates to be identified through other metadata. The additional constraints provide an unambiguous mechanism for identifying the vertical dimension of each data variable while relying on CF metadata to interpret its vertical reference.

The CF metadata associated with the vertical coordinate defines the meaning and reference of the coordinate, including whether it represents quantities such as height, depth, pressure, or a parameterized vertical coordinate.

## Consequences

Source data must use a data model and format capable of preserving coordinate variables and their associated attributes.

Data that does not provide the required CF metadata, or represents vertical information using a different structure, must be normalized before it can be processed as valid source data.

The stricter representation reduces flexibility compared with the CF Conventions but allows vertical dimensions to be identified deterministically without relying on variable names or heuristics.
