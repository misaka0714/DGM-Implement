# DGM-Implement

This repository contains a standalone C++17 implementation of the algorithms
introduced in **Cross-Model Index Migration for Approximate Nearest Neighbor
Search**. It is intended for reading, integration, and extension of DGM itself.
It does not contain dataset loaders, benchmark drivers, paper-specific
configurations, generated indexes, or experiment results.

## Problem Background and Intuition

A graph ANN index records more than a vector array. Its edges were selected
using the distances, neighborhood membership, and navigation cues induced by
the embedding model available at construction time. Re-encoding the corpus
with another model changes those geometric relations. Replacing only the
stored vectors therefore leaves search evaluating new-model distances while
traversing edges chosen in the old space, which can lead to stale
neighborhoods and false local optima.

DGM is motivated by an empirical property of the old topology that we call
**residual reachability**. The old and new embedding spaces need not be
coordinate-aligned or geometrically compatible, but both models encode the
same ID-paired objects and may preserve part of the similarity relations among
them. The old graph materializes these relations not only as direct edges but
also as multi-hop connectivity. Consequently, an exact new-model neighbor may
disappear from a node's old adjacency list yet remain reachable through a
short path in the old graph.

Residual reachability concerns the availability of useful candidates, not the
validity of inherited edges under the new metric. Old-model rankings and
construction geometry can both degrade after a model switch, so DGM treats
the old topology only as a candidate-discovery substrate. Exact new-model
distances validate the candidates and determine every output adjacency list.
This separation provides the central intuition for graph-index migration: reuse
short-path structure instead of rediscovering all candidate relationships, but
never assume that the old graph is already correct for the new embedding
space.

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

## Preparing Datasets and Embeddings

DGM does not redistribute datasets or model checkpoints. The following are the
source pages and checkpoint identifiers used for the paper workloads. Hugging
Face datasets can be obtained with `datasets.load_dataset`, and linked model
identifiers can be passed to the loading API documented on each model card.
Some providers may require authentication or acceptance of their terms.

| Workload | Dataset source | Old encoder -> new encoder |
| --- | --- | --- |
| CIFAR-10 | [`torchvision.datasets.CIFAR10`](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) with `download=True` | [`torchvision.models.resnet18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html) -> [`torchvision.models.resnet34`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet34.html), both with `IMAGENET1K_V1` weights |
| AG News | [`fancyzhx/ag_news`](https://huggingface.co/datasets/fancyzhx/ag_news) | [`nvidia/NV-Embed-v2`](https://huggingface.co/nvidia/NV-Embed-v2) -> [`Salesforce/SFR-Embedding-Mistral`](https://huggingface.co/Salesforce/SFR-Embedding-Mistral) |
| Yelp | [`Yelp/yelp_review_full`](https://huggingface.co/datasets/Yelp/yelp_review_full) | [`sentence-transformers/all-distilroberta-v1`](https://huggingface.co/sentence-transformers/all-distilroberta-v1) -> [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) |
| Amazon-1536 | [`fancyzhx/amazon_polarity`](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | [`Qwen/Qwen2-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct) -> [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| FineWeb | [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT` configuration | [`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1) -> [`sentence-transformers/paraphrase-MiniLM-L3-v2`](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2) |
| Yahoo | [`community-datasets/yahoo_answers_topics`](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) -> [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) |
| QuickDraw | [official Quick, Draw! bitmap files](https://github.com/googlecreativelab/quickdraw-dataset) | [`facebook/metaclip-b32-400m`](https://huggingface.co/facebook/metaclip-b32-400m) -> [`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) |
| MS MARCO | [official MS MARCO](https://microsoft.github.io/msmarco/) or the [`Tevatron/msmarco-passage-corpus`](https://huggingface.co/datasets/Tevatron/msmarco-passage-corpus) passage loader | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) -> [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) |
| C4 English (scale study) | [`allenai/c4`](https://huggingface.co/datasets/allenai/c4), `en` configuration | [`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2) -> [`thenlper/gte-base`](https://huggingface.co/thenlper/gte-base) |

Prepare an old graph and its replacement embedding matrix as follows:

1. Fix an immutable sequence of object IDs. If a workload is sampled, select
   the IDs once and reuse exactly that sequence for both encoders.
2. Encode every object with the old and new checkpoints in evaluation mode.
   Follow each model card's own text prefix, tokenization, pooling,
   normalization, image transform, and feature-extraction instructions; these
   details are model-specific and should not be replaced with one generic
   pooling rule.
3. Build or load the input graph using the old embeddings. Store the new
   embeddings as a contiguous, row-major `float32` matrix in the same object-ID
   order, then pass that matrix to `dgm::EmbeddingSpace`.
4. Select the distance metric used for the target embedding space. Explicit
   L2 normalization is optional for `dgm::Metric::kCosine` because the library
   computes vector norms, although model-prescribed normalization should still
   be retained.

The graph node count must equal the number of target rows, and node IDs must
map to the same objects in both versions. The old and new embedding dimensions
need not match: only the target dimension is supplied to `EmbeddingSpace`.
For traceability, retain the dataset revision and split, selected object IDs,
checkpoint revisions, input formatting, pooling method, normalization choice,
and target metric. Dataset and model licenses remain those of their respective
providers.

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
