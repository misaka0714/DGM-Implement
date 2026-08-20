// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dgm/embedding_space.hpp"

namespace dgm::internal {

class PackedSignatures {
public:
    PackedSignatures(const EmbeddingSpace& space,
                     std::size_t calibration_sample_size,
                     std::uint64_t random_seed,
                     std::size_t thread_count);

    [[nodiscard]] std::int32_t agreement(NodeId lhs, NodeId rhs) const noexcept;

private:
    std::size_t word_count_{0};
    std::uint64_t tail_mask_{0};
    std::vector<std::uint64_t> words_;
};

}  // namespace dgm::internal
