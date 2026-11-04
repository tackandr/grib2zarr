# ADR 0001: One grid and CRS interpretation

- Status: Accepted
- Date: 2026-08-19
- Deciders: Andreas Tack

## Context

The GRIB-to-Zarr conversion tool must produce datasets with consistent, unambiguous spatial metadata that can be interpreted according to the CF Conventions.

CF provides mechanisms for describing coordinate reference systems (CRS), spatial coordinates, and the relationship between data variables and grid mappings. However, CF allows multiple ways to represent spatial information, and not all metadata required for unambiguous identification of spatial dimensions is mandatory.

In particular, an xarray Dataset may contain multiple grid mappings, multiple spatial grids, or coordinate variables without explicit axis identification. Supporting these cases would require additional logic to determine which coordinates and CRS apply to each spatial data variable.

To simplify spatial metadata interpretation and ensure consistent output, the conversion tool requires a single, explicitly defined spatial grid and CRS for each dataset.

## Decision

The conversion tool adopts a constrained interpretation of CF spatial metadata in which each dataset has exactly one spatial grid and one coordinate reference system.

The CRS MUST be represented by a single CF grid-mapping variable referenced by all spatial data variables. The horizontal dimensions MUST be explicitly identifiable through one-dimensional coordinate variables carrying the CF axis attributes.

Datasets that do not satisfy these requirements MUST be rejected as invalid spatial source data.

### Rules

Valid spatial source data MUST satisfy the following requirements:

- The Dataset contains exactly one CF grid-mapping variable describing the coordinate reference system. The grid-mapping variable MUST be scalar and MUST be included as an xarray coordinate.
- Spatial data variables MUST reference this grid-mapping variable through their CF grid_mapping attribute.
- The Dataset contains exactly two spatial dimensions.
- Each spatial dimension MUST have an associated one-dimensional coordinate variable.
- Exactly one of these coordinate variables MUST have the attribute axis="X", and exactly one MUST have axis="Y".
- No other coordinate variable in the Dataset may declare axis="X" or axis="Y".

These requirements are stricter than the CF Conventions. In particular, CF does not require the axis attribute. The additional constraints provide an unambiguous mechanism for identifying the horizontal dimensions and coordinate reference system when deriving the spatial reference of an xarray dataset.


## Consequences

### Positive

- Spatial dimensions and the CRS can be identified deterministically without relying on coordinate names, dimension ordering, or other heuristics.
- All spatial data variables share the same grid and CRS, simplifying metadata conversion and downstream processing.
- The conversion tool can use a single, consistent approach to interpreting spatial metadata.
- Invalid or ambiguous spatial metadata can be detected early through explicit validation.

### Negative

- Some datasets that are valid according to the CF Conventions will not satisfy these requirements.
- Datasets containing multiple spatial grids or coordinate reference systems cannot be represented without splitting them into separate datasets.
- Source datasets may require normalization or additional metadata, particularly explicit axis attributes, before they can be processed.
- The constraints limit support for more complex CF spatial representations.
