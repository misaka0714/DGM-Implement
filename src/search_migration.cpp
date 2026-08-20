// SPDX-License-Identifier: Apache-2.0

#include "dgm/migrate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "candidate.hpp"
#include "neighbor_selection.hpp"
#include "parallel.hpp"

namespace dgm {
namespace {

struct SearchScratch {
    SearchScratch(std::size_t node_count, std::size_t beam_width, std::size_t max_degree)
        : depth_plus_one(node_count, 0) {
        retained.reserve(beam_width);
        touched.reserve(1024);
        expanded_neighbors.reserve(max_degree);
        result_heap.reserve(beam_width);
        frontier_heap.reserve(beam_width);
        sorted_results.reserve(beam_width);
    }

    std::vector<std::uint8_t> depth_plus_one;
    std::vector<NodeId> touched;
    std::unordered_set<NodeId> retained;
    std::vector<NodeId> expanded_neighbors;
    std::vector<internal::Candidate> result_heap;
    std::vector<internal::Candidate> frontier_heap;
    std::vector<internal::Candidate> sorted_results;
    std::uint64_t nodes_processed{0};
    std::uint64_t adjacency_expansions{0};
    std::uint64_t source_distance_evaluations{0};
    std::uint64_t pruning_distance_evaluations{0};
};

void
ValidateInputs(const Graph& graph, const EmbeddingSpace& space, const SearchConfig& config) {
    if (graph.size() != space.size()) {
        throw std::invalid_argument("graph and embedding row counts must match");
    }
    if (config.max_degree == 0 || config.beam_width == 0 || config.max_hops == 0 ||
        config.work_block_size == 0) {
        throw std::invalid_argument("DGM-Search size parameters must be positive");
    }
    if (config.beam_width < config.max_degree) {
        throw std::invalid_argument("DGM-Search beam width must be at least max degree");
    }
    if (config.max_hops > 254) {
        throw std::invalid_argument("DGM-Search supports at most 254 hops");
    }
    if (!(config.diversity_factor > 0.0F) || !std::isfinite(config.diversity_factor)) {
        throw std::invalid_argument("DGM-Search diversity factor must be finite and positive");
    }
    if (!graph.empty() && config.entry_point >= graph.size()) {
        throw std::invalid_argument("DGM-Search entry point is outside the graph");
    }
}

}  // namespace

MigrationResult
RunDgmSearch(const Graph& starting_graph,
             const EmbeddingSpace& new_space,
             const SearchConfig& config) {
    ValidateInputs(starting_graph, new_space, config);
    const auto started = std::chrono::steady_clock::now();
    if (starting_graph.empty()) {
        return {Graph{}, MigrationStats{}};
    }

    Graph graph = starting_graph;
    auto locks = std::make_unique<std::shared_mutex[]>(graph.size());
    const std::size_t worker_count = internal::ResolveThreadCount(
        config.thread_count, graph.size(), config.work_block_size);
    std::vector<std::unique_ptr<SearchScratch>> scratch;
    scratch.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        scratch.push_back(
            std::make_unique<SearchScratch>(graph.size(), config.beam_width, config.max_degree));
    }

    internal::ParallelFor(
        graph.size(),
        config.thread_count,
        config.work_block_size,
        [&](std::size_t worker_id, std::size_t begin, std::size_t end) {
            SearchScratch& local = *scratch[worker_id];
            for (std::size_t source_index = begin; source_index < end; ++source_index) {
                const NodeId source = static_cast<NodeId>(source_index);
                local.touched.clear();
                local.retained.clear();
                local.expanded_neighbors.clear();
                local.result_heap.clear();
                local.frontier_heap.clear();
                local.sorted_results.clear();

                const auto mark_seen = [&](NodeId id, std::size_t depth) {
                    if (local.depth_plus_one[id] != 0) {
                        return false;
                    }
                    local.depth_plus_one[id] = static_cast<std::uint8_t>(depth + 1);
                    local.touched.push_back(id);
                    return true;
                };
                mark_seen(source, 0);

                const auto add_to_beam = [&](const internal::Candidate& candidate,
                                             std::size_t depth) {
                    bool retained = false;
                    if (local.result_heap.size() < config.beam_width) {
                        local.result_heap.push_back(candidate);
                        std::push_heap(local.result_heap.begin(),
                                       local.result_heap.end(),
                                       internal::CandidateLessComparator{});
                        local.retained.insert(candidate.id);
                        retained = true;
                    } else if (internal::CandidateLess(candidate, local.result_heap.front())) {
                        std::pop_heap(local.result_heap.begin(),
                                      local.result_heap.end(),
                                      internal::CandidateLessComparator{});
                        const NodeId evicted = local.result_heap.back().id;
                        local.result_heap.pop_back();
                        local.retained.erase(evicted);
                        local.result_heap.push_back(candidate);
                        std::push_heap(local.result_heap.begin(),
                                       local.result_heap.end(),
                                       internal::CandidateLessComparator{});
                        local.retained.insert(candidate.id);
                        retained = true;
                    }
                    if (retained && depth < config.max_hops) {
                        local.frontier_heap.push_back(candidate);
                        std::push_heap(local.frontier_heap.begin(),
                                       local.frontier_heap.end(),
                                       internal::CandidateGreaterComparator{});
                    }
                };

                {
                    std::shared_lock<std::shared_mutex> lock(locks[source]);
                    local.expanded_neighbors = graph.adjacency_[source];
                }
                ++local.adjacency_expansions;
                for (const NodeId neighbor : local.expanded_neighbors) {
                    if (!mark_seen(neighbor, 1)) {
                        continue;
                    }
                    add_to_beam({new_space.distance(source, neighbor), neighbor}, 1);
                    ++local.source_distance_evaluations;
                }

                if (local.result_heap.empty() && config.entry_point != source &&
                    mark_seen(config.entry_point, 0)) {
                    add_to_beam(
                        {new_space.distance(source, config.entry_point), config.entry_point}, 0);
                    ++local.source_distance_evaluations;
                }

                while (!local.frontier_heap.empty()) {
                    std::pop_heap(local.frontier_heap.begin(),
                                  local.frontier_heap.end(),
                                  internal::CandidateGreaterComparator{});
                    const internal::Candidate current = local.frontier_heap.back();
                    local.frontier_heap.pop_back();
                    if (local.retained.find(current.id) == local.retained.end()) {
                        continue;
                    }

                    const std::size_t current_depth = local.depth_plus_one[current.id] - 1;
                    if (current_depth >= config.max_hops) {
                        continue;
                    }
                    {
                        std::shared_lock<std::shared_mutex> lock(locks[current.id]);
                        local.expanded_neighbors = graph.adjacency_[current.id];
                    }
                    ++local.adjacency_expansions;
                    const std::size_t next_depth = current_depth + 1;
                    for (const NodeId candidate_id : local.expanded_neighbors) {
                        if (!mark_seen(candidate_id, next_depth)) {
                            continue;
                        }
                        add_to_beam(
                            {new_space.distance(source, candidate_id), candidate_id}, next_depth);
                        ++local.source_distance_evaluations;
                    }
                }

                for (const NodeId id : local.touched) {
                    local.depth_plus_one[id] = 0;
                }

                local.sorted_results = local.result_heap;
                std::sort(local.sorted_results.begin(),
                          local.sorted_results.end(),
                          internal::CandidateLess);
                auto selected = internal::SelectDiverseNeighbors(
                    local.sorted_results,
                    config.max_degree,
                    config.diversity_factor,
                    new_space,
                    &local.pruning_distance_evaluations);
                {
                    std::unique_lock<std::shared_mutex> lock(locks[source]);
                    graph.adjacency_[source] = std::move(selected);
                }
                ++local.nodes_processed;
            }
        });

    MigrationStats stats;
    for (const auto& worker : scratch) {
        stats.nodes_processed += worker->nodes_processed;
        stats.adjacency_expansions += worker->adjacency_expansions;
        stats.source_distance_evaluations += worker->source_distance_evaluations;
        stats.pruning_distance_evaluations += worker->pruning_distance_evaluations;
    }
    stats.elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return {std::move(graph), stats};
}

}  // namespace dgm
