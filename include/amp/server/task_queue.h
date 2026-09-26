// Single-worker task queue.
//
// This exists because a llama_context is not reentrant. Two generations on one context corrupt the
// KV cache and the sampler's state, and in practice the process dies inside ggml_abort. Agentic
// clients make that unavoidable rather than theoretical: OpenCode issues a second, small request
// (conversation title, summary) while the main stream is still running, on a different connection.
//
// llama.cpp solves this the same way - server_queue holds a deque of tasks and one worker thread
// drains it (with --parallel N it hands tasks to N *sequences*, but still decodes one task at a
// time). A request therefore waits its turn instead of corrupting the context, and the slot/prefix
// cache only ever sees one task at a time, which is what makes it correct rather than merely
// unlikely to break.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace amp {

class TaskQueue {
public:
    using Task = std::function<void()>;

    TaskQueue();
    ~TaskQueue();

    TaskQueue(const TaskQueue &) = delete;
    TaskQueue & operator=(const TaskQueue &) = delete;

    // Enqueue for execution on the worker thread. Never blocks.
    void post(Task task);

    // Ask the worker to finish after the tasks already queued. Queued tasks are still run.
    void stop();

    // Block until everything queued so far has run. Used by tests and by shutdown.
    void drain();

    struct Stats {
        uint64_t posted   = 0;
        uint64_t done     = 0;
        uint64_t failed   = 0;   // tasks that threw
        uint64_t in_flight = 0;
        uint64_t depth    = 0;   // queued, not yet started
    };
    Stats stats() const;
    std::string describe() const;

private:
    void work();

    mutable std::mutex      mu_;
    std::condition_variable cv_task_;   // worker waits here
    std::condition_variable cv_idle_;   // drain() waits here
    std::deque<Task>        tasks_;
    std::thread             worker_;
    bool                    stop_     = false;
    bool                    busy_     = false;
    uint64_t                posted_   = 0;
    uint64_t                done_     = 0;
    uint64_t                failed_   = 0;
};

} // namespace amp
