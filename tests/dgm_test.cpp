// SPDX-License-Identifier: Apache-2.0

#include <dgm/dgm.hpp>

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

void
Check(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void
TestEmbeddingDistances() {
    const std::vector<float> vectors{1.0F, 0.0F, 0.0F, 1.0F};
    const dgm::EmbeddingSpace cosine(vectors.data(), 2, 2, dgm::Metric::kCosine, 1);
    Check(std::abs(cosine.distance(0, 1) - 1.0F) < 1.0e-6F,
          "orthogonal cosine vectors should have distance one");

    const dgm::EmbeddingSpace l2(vectors.data(), 2, 2, dgm::Metric::kSquaredL2, 1);
    Check(std::abs(l2.distance(0, 1) - 2.0F) < 1.0e-6F,
          "squared-L2 distance is incorrect");
}

void
TestGraphCsrRoundTrip() {
    const std::vector<std::uint64_t> offsets{0, 2, 3, 4};
    const std::vector<dgm::NodeId> neighbors{1, 2, 2, 0};
    const dgm::Graph graph = dgm::Graph::FromCsr(offsets, neighbors);
    Check(graph.csr_offsets() == offsets, "CSR offsets did not round-trip");
    Check(graph.csr_neighbors() == neighbors, "CSR neighbors did not round-trip");
}

void
TestLocalFindsTwoHopNeighbor() {
    const dgm::Graph graph({{1}, {0, 2}, {1, 3}, {2}});
    const std::vector<float> vectors{0.0F, 10.0F, 0.1F, 20.0F};
    const dgm::EmbeddingSpace space(
        vectors.data(), graph.size(), 1, dgm::Metric::kSquaredL2, 1);

    dgm::LocalConfig config;
    config.max_degree = 1;
    config.shortlist_size = 1;
    config.calibration_sample_size = graph.size();
    config.thread_count = 2;
    config.work_block_size = 1;
    const auto migrated = dgm::RunDgmLocal(graph, space, config);

    Check(migrated.graph.neighbors(0) == dgm::Graph::AdjacencyList{2},
          "DGM-Local did not replace a stale edge with the closer two-hop node");
    Check(migrated.stats.nodes_processed == graph.size(),
          "DGM-Local node counter is incorrect");
}

void
TestLocalIsDeterministicAcrossThreadCounts() {
    const dgm::Graph graph({{1, 2}, {0, 3}, {0, 3}, {1, 2}});
    const std::vector<float> vectors{0.0F, 4.0F, 1.0F, 3.0F};
    const dgm::EmbeddingSpace space(
        vectors.data(), graph.size(), 1, dgm::Metric::kSquaredL2, 1);

    dgm::LocalConfig serial_config;
    serial_config.max_degree = 2;
    serial_config.shortlist_size = 2;
    serial_config.calibration_sample_size = graph.size();
    serial_config.thread_count = 1;
    dgm::LocalConfig parallel_config = serial_config;
    parallel_config.thread_count = 4;
    parallel_config.work_block_size = 1;

    const auto serial = dgm::RunDgmLocal(graph, space, serial_config);
    const auto parallel = dgm::RunDgmLocal(graph, space, parallel_config);
    for (dgm::NodeId id = 0; id < graph.size(); ++id) {
        Check(serial.graph.neighbors(id) == parallel.graph.neighbors(id),
              "DGM-Local changed with worker count");
    }
}

void
TestSearchHonorsHopBoundary() {
    const dgm::Graph graph({{1}, {0, 2}, {1, 3}, {2}});
    const std::vector<float> vectors{0.0F, 10.0F, 9.0F, 0.1F};
    const dgm::EmbeddingSpace space(
        vectors.data(), graph.size(), 1, dgm::Metric::kSquaredL2, 1);

    dgm::SearchConfig shallow_config;
    shallow_config.max_degree = 1;
    shallow_config.beam_width = 1;
    shallow_config.max_hops = 2;
    shallow_config.thread_count = 1;
    const auto shallow = dgm::RunDgmSearch(graph, space, shallow_config);
    Check(shallow.graph.neighbors(0) == dgm::Graph::AdjacencyList{2},
          "two-hop search should stop before expanding node 2");

    dgm::SearchConfig deep_config = shallow_config;
    deep_config.max_hops = 3;
    const auto deep = dgm::RunDgmSearch(graph, space, deep_config);
    Check(deep.graph.neighbors(0) == dgm::Graph::AdjacencyList{3},
          "three-hop search did not retain the close depth-three node");
}

void
TestInputValidation() {
    const dgm::Graph graph({{1}, {0}});
    const std::vector<float> one_vector{1.0F};
    const dgm::EmbeddingSpace space(
        one_vector.data(), 1, 1, dgm::Metric::kSquaredL2, 1);
    bool threw = false;
    try {
        static_cast<void>(dgm::RunDgmLocal(graph, space));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    Check(threw, "row-count mismatch should be rejected");
}

void
TestParallelMigrationMaintainsGraphInvariants() {
    constexpr std::size_t node_count = 128;
    constexpr std::size_t dimensions = 8;
    std::vector<dgm::Graph::AdjacencyList> adjacency(node_count);
    for (std::size_t id = 0; id < node_count; ++id) {
        adjacency[id] = {
            static_cast<dgm::NodeId>((id + 1) % node_count),
            static_cast<dgm::NodeId>((id + 7) % node_count),
            static_cast<dgm::NodeId>((id + node_count - 1) % node_count),
            static_cast<dgm::NodeId>((id + node_count - 7) % node_count),
        };
    }
    const dgm::Graph graph(std::move(adjacency));
    std::vector<float> vectors(node_count * dimensions);
    for (std::size_t id = 0; id < node_count; ++id) {
        for (std::size_t dim = 0; dim < dimensions; ++dim) {
            vectors[id * dimensions + dim] =
                static_cast<float>(((id + 3) * (dim + 5)) % 97 + 1);
        }
    }
    const dgm::EmbeddingSpace space(
        vectors.data(), node_count, dimensions, dgm::Metric::kCosine, 4);

    dgm::LocalConfig local_config;
    local_config.max_degree = 8;
    local_config.shortlist_size = 24;
    local_config.calibration_sample_size = 32;
    local_config.thread_count = 8;
    local_config.work_block_size = 2;
    const auto local = dgm::RunDgmLocal(graph, space, local_config);

    dgm::SearchConfig search_config;
    search_config.max_degree = 8;
    search_config.beam_width = 24;
    search_config.max_hops = 3;
    search_config.thread_count = 8;
    search_config.work_block_size = 2;
    const auto refined = dgm::RunDgmSearch(local.graph, space, search_config);

    for (dgm::NodeId source = 0; source < refined.graph.size(); ++source) {
        const auto& row = refined.graph.neighbors(source);
        Check(row.size() <= search_config.max_degree, "migrated row exceeds degree bound");
        std::unordered_set<dgm::NodeId> unique;
        for (const dgm::NodeId target : row) {
            Check(target < refined.graph.size(), "migrated row contains an invalid ID");
            Check(target != source, "migrated row contains a self edge");
            Check(unique.insert(target).second, "migrated row contains a duplicate edge");
        }
    }
}

}  // namespace

int
main() {
    try {
        TestEmbeddingDistances();
        TestGraphCsrRoundTrip();
        TestLocalFindsTwoHopNeighbor();
        TestLocalIsDeterministicAcrossThreadCounts();
        TestSearchHonorsHopBoundary();
        TestInputValidation();
        TestParallelMigrationMaintainsGraphInvariants();
    } catch (const std::exception& error) {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "All DGM tests passed.\n";
    return 0;
}
