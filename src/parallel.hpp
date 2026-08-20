// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace dgm::internal {

inline std::size_t
ResolveThreadCount(std::size_t requested, std::size_t item_count, std::size_t block_size) {
    if (item_count == 0) {
        return 0;
    }
    const std::size_t hardware = std::max<std::size_t>(1, std::thread::hardware_concurrency());
    const std::size_t desired = requested == 0 ? hardware : requested;
    const std::size_t block_count = (item_count + block_size - 1) / block_size;
    return std::max<std::size_t>(1, std::min(desired, block_count));
}

template <typename Worker>
void
ParallelFor(std::size_t item_count,
            std::size_t requested_threads,
            std::size_t block_size,
            Worker&& worker) {
    block_size = std::max<std::size_t>(1, block_size);
    const std::size_t thread_count =
        ResolveThreadCount(requested_threads, item_count, block_size);
    if (thread_count == 0) {
        return;
    }
    if (thread_count == 1) {
        worker(0, 0, item_count);
        return;
    }

    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr first_error;
    std::mutex error_mutex;
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (std::size_t worker_id = 0; worker_id < thread_count; ++worker_id) {
        threads.emplace_back([&, worker_id]() {
            try {
                while (!stop.load(std::memory_order_relaxed)) {
                    const std::size_t begin =
                        next.fetch_add(block_size, std::memory_order_relaxed);
                    if (begin >= item_count) {
                        break;
                    }
                    worker(worker_id, begin, std::min(begin + block_size, item_count));
                }
            } catch (...) {
                stop.store(true, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (!first_error) {
                    first_error = std::current_exception();
                }
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    if (first_error) {
        std::rethrow_exception(first_error);
    }
}

}  // namespace dgm::internal
