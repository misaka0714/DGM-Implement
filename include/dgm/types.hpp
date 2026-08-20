// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>

namespace dgm {

using NodeId = std::uint32_t;

enum class Metric {
    kCosine,
    kSquaredL2,
};

struct MigrationStats {
    std::uint64_t nodes_processed{0};
    std::uint64_t candidates_screened{0};
    std::uint64_t adjacency_expansions{0};
    std::uint64_t source_distance_evaluations{0};
    std::uint64_t pruning_distance_evaluations{0};
    double elapsed_seconds{0.0};
};

}  // namespace dgm
