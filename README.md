# DGM Algorithm

This repository contains a standalone C++17 implementation of the algorithms
introduced in **Cross-Model Index Migration for Approximate Nearest Neighbor
Search**. It is intended for reading, integration, and extension of DGM itself.
It does not contain dataset loaders, benchmark drivers, paper-specific
configurations, generated indexes, or experiment results.

## Why graph migration?

A graph ANN index records more than a vector array. Its edges were selected
using the distances and neighborhood ordering induced by the embedding model
available at construction time. Re-encoding the corpus with another model and
replacing only the vectors leaves search traversing stale edges under a new
distance function.

The old topology is nevertheless useful: new-model neighbors often remain
reachable through short paths even when individual edges and rankings have
changed. DGM uses that residual reachability to discover candidates and uses
exact new-model distances to determine every output adjacency list.

## Algorithms

- **DGM-Local** reads an immutable input graph, forms a deduplicated two-hop
  pool for each node, shortlists candidates with packed new-space sign
  signatures, restores every inherited one-hop neighbor to the exact candidate
  set, and performs exact distance ordering and diversity-aware selection.
- **DGM-Search** starts from either a vector-replacement graph or a DGM-Local
  graph. For each node it runs a new-space beam traversal whose width and hop
  depth are bounded independently, then selects a new adjacency list from the
  retained candidates.

Both algorithms operate on one graph layer. A hierarchical index can preserve
its existing entry point and layer membership while invoking the migration
operation independently for each stored layer.

## Public API

The library has three input types:

- `dgm::Graph`: an in-memory adjacency list, constructible directly or from
  CSR arrays;
- `dgm::EmbeddingSpace`: a non-owning row-major `float32` matrix with cosine
  or squared-L2 distance; and
- `dgm::LocalConfig` / `dgm::SearchConfig`: typed algorithm controls.

The embedding row index and graph node ID must refer to the same object.
The embedding buffer must remain alive while `EmbeddingSpace` is in use.

```cpp
#include <dgm/dgm.hpp>

std::vector<dgm::Graph::AdjacencyList> adjacency = LoadAdjacency();
std::vector<float> target_vectors = LoadTargetEmbeddings();

dgm::Graph old_graph(std::move(adjacency));
dgm::EmbeddingSpace target_space(
    target_vectors.data(), old_graph.size(), dimensions, dgm::Metric::kCosine);

dgm::LocalConfig local_config;
auto local = dgm::RunDgmLocal(old_graph, target_space, local_config);

dgm::SearchConfig search_config;
auto refined = dgm::RunDgmSearch(local.graph, target_space, search_config);
```

`MigrationResult` contains the migrated graph and lightweight work counters.
DGM-Local is snapshot-based and deterministic for fixed inputs and seed.
DGM-Search follows the paper's asynchronous update semantics: workers may read
lists repaired earlier in the same pass, so parallel execution order can affect
the final graph.

## Build

The implementation has no third-party runtime dependency.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run the small in-memory example with:

```bash
./build/dgm_minimal
```

## Layout

```text
include/dgm/              Public API
src/local_migration.cpp   DGM-Local
src/search_migration.cpp  DGM-Search
src/packed_signatures.cpp Packed sign screening
src/neighbor_selection.cpp Exact diversity-aware selection
examples/minimal.cpp      In-memory integration example
tests/dgm_test.cpp        Algorithm and API tests
```

## Scope

This package deliberately leaves index serialization, vector storage, query
serving, and hierarchical-layer orchestration to the host ANN system. That
keeps the algorithm boundary explicit and avoids coupling DGM to an experiment
file format or a particular graph-index implementation.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
