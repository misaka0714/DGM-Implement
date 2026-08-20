// SPDX-License-Identifier: Apache-2.0

#include "dgm/embedding_space.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "parallel.hpp"

namespace dgm {

EmbeddingSpace::EmbeddingSpace(const float* values,
                               std::size_t row_count,
                               std::size_t dimensions,
                               Metric metric,
                               std::size_t thread_count)
    : values_(values), row_count_(row_count), dimensions_(dimensions), metric_(metric) {
    if (values_ == nullptr && row_count_ != 0) {
        throw std::invalid_argument("embedding data is null");
    }
    if (dimensions_ == 0 && row_count_ != 0) {
        throw std::invalid_argument("embedding dimension must be positive");
    }
    if (row_count_ != 0 && dimensions_ > std::numeric_limits<std::size_t>::max() / row_count_) {
        throw std::invalid_argument("embedding matrix dimensions overflow addressable size");
    }
    if (row_count_ > static_cast<std::size_t>(std::numeric_limits<NodeId>::max())) {
        throw std::invalid_argument("embedding matrix has more rows than NodeId can represent");
    }
    if (metric_ != Metric::kCosine && metric_ != Metric::kSquaredL2) {
        throw std::invalid_argument("unsupported distance metric");
    }

    if (metric_ == Metric::kCosine) {
        inverse_norms_.resize(row_count_);
    }
    internal::ParallelFor(
        row_count_, thread_count, 256, [&](std::size_t, std::size_t begin, std::size_t end) {
            for (std::size_t id = begin; id < end; ++id) {
                const float* vector = values_ + id * dimensions_;
                double squared_norm = 0.0;
                for (std::size_t dim = 0; dim < dimensions_; ++dim) {
                    const float value = vector[dim];
                    if (!std::isfinite(value)) {
                        throw std::invalid_argument("embedding matrix contains a non-finite value");
                    }
                    squared_norm += static_cast<double>(value) * value;
                }
                if (metric_ == Metric::kCosine && !(squared_norm > 0.0)) {
                    throw std::invalid_argument("cosine embeddings must have nonzero norm");
                }
                if (metric_ == Metric::kCosine) {
                    inverse_norms_[id] = static_cast<float>(1.0 / std::sqrt(squared_norm));
                }
            }
        });
}

std::size_t
EmbeddingSpace::size() const noexcept {
    return row_count_;
}

std::size_t
EmbeddingSpace::dimensions() const noexcept {
    return dimensions_;
}

Metric
EmbeddingSpace::metric() const noexcept {
    return metric_;
}

const float*
EmbeddingSpace::row(NodeId id) const {
    if (id >= row_count_) {
        throw std::out_of_range("embedding row is out of range");
    }
    return values_ + static_cast<std::size_t>(id) * dimensions_;
}

float
EmbeddingSpace::distance(NodeId lhs, NodeId rhs) const {
    const float* lhs_vector = row(lhs);
    const float* rhs_vector = row(rhs);
    double accumulator = 0.0;
    if (metric_ == Metric::kSquaredL2) {
        for (std::size_t dim = 0; dim < dimensions_; ++dim) {
            const double difference =
                static_cast<double>(lhs_vector[dim]) - rhs_vector[dim];
            accumulator += difference * difference;
        }
        return static_cast<float>(accumulator);
    }

    for (std::size_t dim = 0; dim < dimensions_; ++dim) {
        accumulator += static_cast<double>(lhs_vector[dim]) * rhs_vector[dim];
    }
    const double similarity = accumulator * inverse_norms_[lhs] * inverse_norms_[rhs];
    return static_cast<float>(1.0 - std::clamp(similarity, -1.0, 1.0));
}

}  // namespace dgm
