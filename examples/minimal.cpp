// SPDX-License-Identifier: Apache-2.0

#include <dgm/dgm.hpp>

#include <iostream>
#include <utility>
#include <vector>

int
main() {
    std::vector<dgm::Graph::AdjacencyList> adjacency{
        {1, 2}, {0, 3}, {0, 4}, {1, 5}, {2, 5}, {3, 4},
    };
    std::vector<float> target_vectors{
        1.00F, 0.00F,
        0.70F, 0.70F,
        0.98F, 0.20F,
        0.00F, 1.00F,
        0.92F, 0.38F,
        0.20F, 0.98F,
    };

    dgm::Graph old_graph(std::move(adjacency));
    dgm::EmbeddingSpace target_space(
        target_vectors.data(), old_graph.size(), 2, dgm::Metric::kCosine);

    dgm::LocalConfig local_config;
    local_config.max_degree = 2;
    local_config.shortlist_size = 4;
    auto local = dgm::RunDgmLocal(old_graph, target_space, local_config);

    dgm::SearchConfig search_config;
    search_config.max_degree = 2;
    search_config.beam_width = 4;
    search_config.max_hops = 3;
    auto refined = dgm::RunDgmSearch(local.graph, target_space, search_config);

    std::cout << "Migrated neighbors of node 0:";
    for (const dgm::NodeId neighbor : refined.graph.neighbors(0)) {
        std::cout << ' ' << neighbor;
    }
    std::cout << '\n';
    return 0;
}
