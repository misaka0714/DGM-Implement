// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "candidate.hpp"
#include "dgm/embedding_space.hpp"

namespace dgm::internal {

std::vector<NodeId> SelectDiverseNeighbors(const std::vector<Candidate>& sorted_candidates,
                                           std::size_t max_degree,
                                           float diversity_factor,
                                           const EmbeddingSpace& space,
                                           std::uint64_t* distance_evaluations);

}  // namespace dgm::internal
