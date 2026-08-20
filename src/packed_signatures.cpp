// SPDX-License-Identifier: Apache-2.0

#include "packed_signatures.hpp"

#include <algorithm>
#include <limits>
#include <random>
#include <stdexcept>

#include "parallel.hpp"

namespace dgm::internal {
namespace {

std::vector<NodeId>
UniformSample(std::size_t population, std::size_t sample_size, std::uint64_t seed) {
    sample_size = std::min(population, sample_size);
    std::vector<NodeId> sample(sample_size);
    for (std::size_t id = 0; id < sample_size; ++id) {
        sample[id] = static_cast<NodeId>(id);
    }
    std::mt19937_64 random(seed);
    for (std::size_t id = sample_size; id < population; ++id) {
        std::uniform_int_distribution<std::size_t> distribution(0, id);
        const std::size_t selected = distribution(random);
        if (selected < sample_size) {
            sample[selected] = static_cast<NodeId>(id);
        }
    }
    return sample;
}

inline std::uint32_t
Popcount(std::uint64_t value) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::uint32_t>(__builtin_popcountll(value));
#else
    std::uint32_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
#endif
}

}  // namespace

PackedSignatures::PackedSignatures(const EmbeddingSpace& space,
                                   std::size_t calibration_sample_size,
                                   std::uint64_t random_seed,
                                   std::size_t thread_count) {
    if (space.size() == 0) {
        return;
    }
    if (calibration_sample_size == 0) {
        throw std::invalid_argument("calibration sample size must be positive");
    }

    const auto sample = UniformSample(space.size(), calibration_sample_size, random_seed);
    std::vector<double> center(space.dimensions(), 0.0);
    for (const NodeId id : sample) {
        const float* vector = space.row(id);
        for (std::size_t dim = 0; dim < space.dimensions(); ++dim) {
            center[dim] += vector[dim];
        }
    }
    const double denominator = static_cast<double>(sample.size());
    for (double& value : center) {
        value /= denominator;
    }

    word_count_ = (space.dimensions() + 63) / 64;
    const std::size_t tail_bits = space.dimensions() % 64;
    tail_mask_ = tail_bits == 0 ? std::numeric_limits<std::uint64_t>::max()
                                : ((std::uint64_t{1} << tail_bits) - 1);
    words_.assign(space.size() * word_count_, 0);

    ParallelFor(space.size(),
                thread_count,
                256,
                [&](std::size_t, std::size_t begin, std::size_t end) {
                    for (std::size_t id = begin; id < end; ++id) {
                        const float* vector = space.row(static_cast<NodeId>(id));
                        std::uint64_t* signature = words_.data() + id * word_count_;
                        for (std::size_t dim = 0; dim < space.dimensions(); ++dim) {
                            if (static_cast<double>(vector[dim]) >= center[dim]) {
                                signature[dim / 64] |= std::uint64_t{1} << (dim % 64);
                            }
                        }
                    }
                });
}

std::int32_t
PackedSignatures::agreement(NodeId lhs, NodeId rhs) const noexcept {
    if (word_count_ == 0) {
        return 0;
    }
    const std::uint64_t* lhs_words = words_.data() + static_cast<std::size_t>(lhs) * word_count_;
    const std::uint64_t* rhs_words = words_.data() + static_cast<std::size_t>(rhs) * word_count_;
    std::uint32_t score = 0;
    for (std::size_t word = 0; word + 1 < word_count_; ++word) {
        score += Popcount(~(lhs_words[word] ^ rhs_words[word]));
    }
    score += Popcount((~(lhs_words[word_count_ - 1] ^ rhs_words[word_count_ - 1])) & tail_mask_);
    return static_cast<std::int32_t>(score);
}

}  // namespace dgm::internal
