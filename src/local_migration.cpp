// SPDX-License-Identifier: Apache-2.0

#include "dgm/migrate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "candidate.hpp"
#include "neighbor_selection.hpp"
#include "packed_signatures.hpp"
#include "parallel.hpp"

namespace dgm {
namespace {

struct SignCandidate {
    std::int32_t agreement;
    NodeId id;
};

bool
BetterSignCandidate(const SignCandidate& lhs, const SignCandidate& rhs) noexcept {
    return lhs.agreement > rhs.agreement ||
           (lhs.agreement == rhs.agreement && lhs.id < rhs.id);
}

struct LocalScratch {
    std::unordered_set<NodeId> seen;
    std::vector<NodeId> inherited;
    std::vector<NodeId> two_hop;
    std::vector<SignCandidate> scored;
    std::vector<internal::Candidate> exact;
    std::uint64_t nodes_processed{0};
    std::uint64_t candidates_screened{0};
    std::uint64_t adjacency_expansions{0};
    std::uint64_t source_distance_evaluations{0};
    std::uint64_t pruning_distance_evaluations{0};
};

void
ValidateInputs(const Graph& graph, const EmbeddingSpace& space, const LocalConfig& config) {
    if (graph.size() != space.size()) {
        throw std::invalid_argument("graph and embedding row counts must match");
    }
    if (config.max_degree == 0 || config.shortlist_size == 0 ||
        config.calibration_sample_size == 0 || config.work_block_size == 0) {
        throw std::invalid_argument("DGM-Local size parameters must be positive");
    }
    if (!(config.diversity_factor > 0.0F) || !std::isfinite(config.diversity_factor)) {
        throw std::invalid_argument("DGM-Local diversity factor must be finite and positive");
    }
}

}  // namespace

MigrationResult
RunDgmLocal(const Graph& old_graph,
            const EmbeddingSpace& new_space,
            const LocalConfig& config) {
    ValidateInputs(old_graph, new_space, config);
    const auto started = std::chrono::steady_clock::now();
    if (old_graph.empty()) {
        return {Graph{}, MigrationStats{}};
    }

    internal::PackedSignatures signatures(new_space,
                                          config.calibration_sample_size,
                                          config.random_seed,
                                          config.thread_count);
    std::vector<Graph::AdjacencyList> output(old_graph.size());

    const std::size_t worker_count = internal::ResolveThreadCount(
        config.thread_count, old_graph.size(), config.work_block_size);
    std::vector<LocalScratch> scratch(worker_count);
    for (auto& worker : scratch) {
        const std::size_t expanded_degree_hint =
            config.max_degree > old_graph.size() / 8
                ? old_graph.size()
                : config.max_degree * 8;
        const std::size_t reserve_hint = std::max<std::size_t>(
            64, std::min<std::size_t>(old_graph.size(), expanded_degree_hint));
        worker.seen.reserve(reserve_hint);
        worker.inherited.reserve(config.max_degree);
        worker.two_hop.reserve(reserve_hint);
        worker.scored.reserve(reserve_hint);
        worker.exact.reserve(config.shortlist_size + config.max_degree);
    }

    internal::ParallelFor(
        old_graph.size(),
        config.thread_count,
        config.work_block_size,
        [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
            LocalScratch& local = scratch[worker_id];
            for (std::size_t source_index = begin; source_index < end; ++source_index) {
                const NodeId source = static_cast<NodeId>(source_index);
                local.seen.clear();
                local.inherited.clear();
                local.two_hop.clear();
                local.scored.clear();
                local.exact.clear();

                local.seen.insert(source);
                for (const NodeId neighbor : old_graph.neighbors(source)) {
                    if (local.seen.insert(neighbor).second) {
                        local.inherited.push_back(neighbor);
                    }
                }
                ++local.adjacency_expansions;

                for (const NodeId neighbor : local.inherited) {
                    for (const NodeId candidate : old_graph.neighbors(neighbor)) {
                        if (local.seen.insert(candidate).second) {
                            local.two_hop.push_back(candidate);
                        }
                    }
                    ++local.adjacency_expansions;
                }

                local.scored.reserve(local.two_hop.size());
                for (const NodeId candidate : local.two_hop) {
                    local.scored.push_back(
                        {signatures.agreement(source, candidate), candidate});
                }
                local.candidates_screened += local.scored.size();

                const std::size_t shortlist_size =
                    std::min(config.shortlist_size, local.scored.size());
                if (shortlist_size != 0) {
                    std::partial_sort(local.scored.begin(),
                                      local.scored.begin() +
                                          static_cast<std::ptrdiff_t>(shortlist_size),
                                      local.scored.end(),
                                      BetterSignCandidate);
                    local.scored.resize(shortlist_size);
                }

                local.exact.reserve(local.inherited.size() + local.scored.size());
                for (const NodeId candidate : local.inherited) {
                    local.exact.push_back({new_space.distance(source, candidate), candidate});
                }
                for (const SignCandidate& candidate : local.scored) {
                    local.exact.push_back(
                        {new_space.distance(source, candidate.id), candidate.id});
                }
                local.source_distance_evaluations += local.exact.size();
                std::sort(local.exact.begin(), local.exact.end(), internal::CandidateLess);

                output[source] = internal::SelectDiverseNeighbors(
                    local.exact,
                    config.max_degree,
                    config.diversity_factor,
                    new_space,
                    &local.pruning_distance_evaluations);
                ++local.nodes_processed;
            }
        });

    MigrationStats stats;
    for (const auto& worker : scratch) {
        stats.nodes_processed += worker.nodes_processed;
        stats.candidates_screened += worker.candidates_screened;
        stats.adjacency_expansions += worker.adjacency_expansions;
        stats.source_distance_evaluations += worker.source_distance_evaluations;
        stats.pruning_distance_evaluations += worker.pruning_distance_evaluations;
    }
    stats.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return {Graph(std::move(output)), stats};
}

}  // namespace dgm
