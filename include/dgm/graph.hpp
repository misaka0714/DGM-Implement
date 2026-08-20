// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dgm/types.hpp"

namespace dgm {

class EmbeddingSpace;
class Graph;
struct MigrationResult;
struct SearchConfig;

MigrationResult RunDgmSearch(const Graph& starting_graph,
                             const EmbeddingSpace& new_space,
                             const SearchConfig& config);

class Graph {
public:
    using AdjacencyList = std::vector<NodeId>;

    Graph() = default;
    explicit Graph(std::vector<AdjacencyList> adjacency);

    static Graph FromCsr(const std::vector<std::uint64_t>& offsets,
                         const std::vector<NodeId>& neighbors);

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] const AdjacencyList& neighbors(NodeId id) const;
    [[nodiscard]] std::vector<std::uint64_t> csr_offsets() const;
    [[nodiscard]] std::vector<NodeId> csr_neighbors() const;

private:
    friend struct MigrationResult;
    friend MigrationResult RunDgmSearch(const Graph&, const EmbeddingSpace&, const SearchConfig&);

    std::vector<AdjacencyList> adjacency_;
};

}  // namespace dgm
