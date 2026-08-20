// SPDX-License-Identifier: Apache-2.0

#include "neighbor_selection.hpp"

#include <algorithm>
#include <stdexcept>

namespace dgm::internal {

std::vector<NodeId>
SelectDiverseNeighbors(const std::vector<Candidate>& sorted_candidates,
                       std::size_t max_degree,
                       float diversity_factor,
                       const EmbeddingSpace& space,
                       std::uint64_t* distance_evaluations) {
    if (max_degree == 0) {
        throw std::invalid_argument("maximum degree must be positive");
    }
    if (!(diversity_factor > 0.0F)) {
        throw std::invalid_argument("diversity factor must be positive");
    }

    std::vector<NodeId> selected;
    selected.reserve(std::min(max_degree, sorted_candidates.size()));

    // Match the graph-construction heuristic: a pool smaller than the degree
    // bound is already bounded and is returned without pairwise pruning.
    if (sorted_candidates.size() < max_degree) {
        for (const Candidate& candidate : sorted_candidates) {
            selected.push_back(candidate.id);
        }
        return selected;
    }

    for (const Candidate& candidate : sorted_candidates) {
        if (selected.size() == max_degree) {
            break;
        }
        bool keep = true;
        for (const NodeId retained : selected) {
            const float pair_distance = space.distance(candidate.id, retained);
            if (distance_evaluations != nullptr) {
                ++*distance_evaluations;
            }
            if (diversity_factor * pair_distance < candidate.distance) {
                keep = false;
                break;
            }
        }
        if (keep) {
            selected.push_back(candidate.id);
        }
    }
    return selected;
}

}  // namespace dgm::internal
