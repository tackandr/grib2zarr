# ADR 0008: Dataset Data Variable Chunking

- Status: Accepted
- Date: 2026-10-19
- Deciders: Andreas Tack

## Context

Parallel and partial processing of multidimensional datasets requires dividing data variables into smaller, independently accessible chunks. Chunking enables data to be distributed across multiple workers and allows lazy loading of only the data required for a particular operation.

Data variables can be chunked along one, several, or all of their dimensions. The optimal chunk layout depends on the processing operation. For example, temporal operations may benefit from chunking primarily along spatial dimensions, whereas spatial operations may benefit from chunking along the temporal dimension.

Consequently, no single chunk layout is optimal for all processing operations.

Chunk size also involves a trade-off between parallelism, memory consumption, and processing overhead. Smaller chunks enable more fine-grained data access and distribution, whereas larger chunks reduce task scheduling overhead.

When using storage formats that support sharding, such as Zarr v3, multiple chunks can be grouped into larger storage objects. Sharding allows relatively small chunks to be retained without requiring a separate storage object for each chunk.

Therefore, both chunking and sharding must remain flexible to accommodate different datasets, storage requirements, and processing operations.

## Decision

We do not impose restrictions on the dimensions along which input data variables may be chunked or sharded. Processing functions must accommodate input data regardless of its chunk layout, subject to available computational resources.

### Rules

The following requirements apply to processing functions that operate on data variables:

- Each input data variable MAY be chunked along ANY of its dimensions.
- Each input data variable MAY be sharded along ANY of its dimensions.
- Data variables within the same dataset MAY have different chunk layouts.
- Processing functions MUST NOT assume that any particular dimension is contained within a single chunk.
- Processing functions MUST NOT produce incorrect results due to the input chunk layout. Functions MAY fail with an explicit error if the operation cannot be performed within available resources.
- Processing functions MAY rechunk input data when necessary to perform an operation.

## Consequences

Processing functions must account for data distributed across chunk boundaries, including cases where variables in the same dataset have different chunk layouts.

Operations requiring data from multiple chunks must use appropriate algorithms, such as chunk-wise reductions, overlapping computations, or rechunking. Operations combining multiple variables may also need to align their chunk boundaries.

Some processing operations may require additional data movement or memory when the input chunk layout is not optimal for the operation. Consequently, processing performance may vary depending on the input chunk layout.

Processing functions are responsible for adapting to the input chunk layout when necessary. They must not silently produce incorrect results because of chunk boundaries; when an operation cannot be performed within available resources, an explicit failure is acceptable.
