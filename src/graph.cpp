// SPDX-License-Identifier: Apache-2.0

#include "dgm/graph.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace dgm {
namespace {

void
ValidateAdjacency(const std::vector<Graph::AdjacencyList>& adjacency) {
    if (adjacency.size() > static_cast<std::size_t>(std::numeric_limits<NodeId>::max())) {
        throw std::invalid_argument("graph has more nodes than NodeId can represent");
    }
    for (std::size_t source = 0; source < adjacency.size(); ++source) {
        for (const NodeId target : adjacency[source]) {
            if (target >= adjacency.size()) {
                throw std::invalid_argument("graph contains an out-of-range neighbor ID");
            }
            if (target == source) {
                throw std::invalid_argument("graph contains a self edge");
            }
        }
    }
}

}  // namespace

Graph::Graph(std::vector<AdjacencyList> adjacency) : adjacency_(std::move(adjacency)) {
    ValidateAdjacency(adjacency_);
}

Graph
Graph::FromCsr(const std::vector<std::uint64_t>& offsets,
               const std::vector<NodeId>& neighbors) {
    if (offsets.empty() || offsets.front() != 0 || offsets.back() != neighbors.size()) {
        throw std::invalid_argument("invalid CSR offsets");
    }
    std::vector<AdjacencyList> adjacency(offsets.size() - 1);
    for (std::size_t id = 0; id + 1 < offsets.size(); ++id) {
        if (offsets[id] > offsets[id + 1] || offsets[id + 1] > neighbors.size()) {
            throw std::invalid_argument("CSR offsets must be monotonic and in range");
        }
        adjacency[id].assign(neighbors.begin() + static_cast<std::ptrdiff_t>(offsets[id]),
                             neighbors.begin() + static_cast<std::ptrdiff_t>(offsets[id + 1]));
    }
    return Graph(std::move(adjacency));
}

std::size_t
Graph::size() const noexcept {
    return adjacency_.size();
}

bool
Graph::empty() const noexcept {
    return adjacency_.empty();
}

const Graph::AdjacencyList&
Graph::neighbors(NodeId id) const {
    if (id >= adjacency_.size()) {
        throw std::out_of_range("node ID is outside the graph");
    }
    return adjacency_[id];
}

std::vector<std::uint64_t>
Graph::csr_offsets() const {
    std::vector<std::uint64_t> offsets(adjacency_.size() + 1, 0);
    for (std::size_t id = 0; id < adjacency_.size(); ++id) {
        offsets[id + 1] = offsets[id] + adjacency_[id].size();
    }
    return offsets;
}

std::vector<NodeId>
Graph::csr_neighbors() const {
    std::size_t edge_count = 0;
    for (const auto& row : adjacency_) {
        edge_count += row.size();
    }
    std::vector<NodeId> neighbors;
    neighbors.reserve(edge_count);
    for (const auto& row : adjacency_) {
        neighbors.insert(neighbors.end(), row.begin(), row.end());
    }
    return neighbors;
}

}  // namespace dgm
