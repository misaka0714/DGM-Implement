// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

#include "dgm/types.hpp"

namespace dgm::internal {

struct Candidate {
    float distance;
    NodeId id;
};

inline bool
CandidateLess(const Candidate& lhs, const Candidate& rhs) noexcept {
    return lhs.distance < rhs.distance ||
           (lhs.distance == rhs.distance && lhs.id < rhs.id);
}

struct CandidateLessComparator {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const noexcept {
        return CandidateLess(lhs, rhs);
    }
};

struct CandidateGreaterComparator {
    bool operator()(const Candidate& lhs, const Candidate& rhs) const noexcept {
        return CandidateLess(rhs, lhs);
    }
};

}  // namespace dgm::internal
