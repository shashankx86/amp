#include "amp/server/task_queue.h"

#include "amp/format.h"
#include "amp/log.h"

#include <exception>

namespace amp {

TaskQueue::TaskQueue() {
    worker_ = std::thread([this] { work(); });
}

TaskQueue::~TaskQueue() {
    stop();
}

void TaskQueue::post(Task task) {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_) {
            // Nobody is left to run it. Run inline rather than silently dropping the request: the
            // caller is blocked waiting for this generation, and a hung request is worse than a
            // rejected one.
            task();
            return;
        }
        tasks_.push_back(std::move(task));
        posted_++;
    }
    cv_task_.notify_one();
}

void TaskQueue::stop() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stop_) {
            return;
        }
        stop_ = true;
    }
    cv_task_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void TaskQueue::drain() {
    std::unique_lock<std::mutex> lock(mu_);
    cv_idle_.wait(lock, [this] { return (tasks_.empty() && !busy_) || stop_; });
}

void TaskQueue::work() {
    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_task_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
            if (tasks_.empty()) {
                if (stop_) {
                    return;
                }
                continue;
            }
            task = std::move(tasks_.front());
            tasks_.pop_front();
            busy_ = true;
        }

        // A task that throws must not take the worker - and with it every other queued request -
        // down. The task's own owner is responsible for reporting the failure to its client; here we
        // only have to survive it and count it.
        try {
            task();
        } catch (const std::exception & e) {
            std::lock_guard<std::mutex> lock(mu_);
            failed_++;
            AMP_ERROR("amp: task threw: ", e.what());
        } catch (...) {
            std::lock_guard<std::mutex> lock(mu_);
            failed_++;
            AMP_ERROR("amp: task threw an unknown exception");
        }

        {
            std::lock_guard<std::mutex> lock(mu_);
            busy_ = false;
            done_++;
        }
        cv_idle_.notify_all();
    }
}

TaskQueue::Stats TaskQueue::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    Stats                        s;
    s.posted    = posted_;
    s.done      = done_;
    s.failed    = failed_;
    s.in_flight = busy_ ? 1 : 0;
    s.depth     = (uint64_t) tasks_.size();
    return s;
}

std::string TaskQueue::describe() const {
    const Stats s = stats();
    return format("tasks posted=%llu done=%llu failed=%llu in_flight=%llu queued=%llu",
                  (unsigned long long) s.posted, (unsigned long long) s.done,
                  (unsigned long long) s.failed, (unsigned long long) s.in_flight,
                  (unsigned long long) s.depth);
}

} // namespace amp
