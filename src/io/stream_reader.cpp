#include "amp/io/stream_reader.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace amp {

const char * to_string(ReadPolicy p) {
    switch (p) {
        case ReadPolicy::kCacheWarm: return "cache-warm";
        case ReadPolicy::kStream:    return "stream";
    }
    return "?";
}

std::string StreamStats::to_string() const {
    return format("submitted=%s completed=%s bytes=%s errors=%s",
                  human_count(submitted).c_str(), human_count(completed).c_str(),
                  human_bytes(bytes).c_str(), human_count(errors).c_str());
}

namespace {

// Thread-pool pread() reader. io_uring will slot in behind IStreamReader later; the
// interface is chosen so that swap is a factory change, not a refactor.
class PoolStreamReader final : public IStreamReader {
public:
    PoolStreamReader(const StreamerConfig & cfg, int fd) : cfg_(cfg), fd_(fd) {}

    ~PoolStreamReader() override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    const char * name() const override { return "pread-pool"; }

    Status init(const StreamerConfig & cfg) override {
        cfg_ = cfg;
        const uint32_t n = std::max(1u, cfg_.threads);
        workers_.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            workers_.emplace_back([this] { loop(); });
        }
        return Status::OK();
    }

    Status submit(const ReadRequest & req) override {
        if (req.range.empty()) {
            return Status::OK();
        }
        if (req.policy == ReadPolicy::kStream && req.dst == nullptr) {
            return Status::Error("stream request without destination buffer");
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stop_) {
                return Status::Error("stream reader stopped");
            }
            if (queue_.size() >= cfg_.queue_depth) {
                return Status::Error("stream queue full");
            }
            if (inflight_bytes_ + req.range.length > cfg_.max_inflight_bytes) {
                return Status::Error("stream inflight byte budget exhausted");
            }
            queue_.push_back(req);
            inflight_bytes_ += req.range.length;
            stats_.submitted++;
        }
        cv_.notify_one();
        return Status::OK();
    }

    Status wait_idle(uint32_t timeout_ms) override {
        std::unique_lock<std::mutex> lock(mu_);
        const bool idle = cv_idle_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return queue_.empty() && in_flight_ == 0;
        });
        return idle ? Status::OK() : Status::Error("stream reader wait_idle timed out");
    }

    StreamStats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }

    uint64_t bytes_inflight() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return inflight_bytes_;
    }

private:
    void loop() {
        for (;;) {
            ReadRequest req;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) {
                    return;
                }
                req = queue_.front();
                queue_.pop_front();
                in_flight_++;
            }

            uint8_t * out = (uint8_t *) req.dst;
            uint64_t  done = 0;
            while (done < req.range.length) {
                const ssize_t n = pread(fd_, out + done, (size_t) (req.range.length - done),
                                        (off_t) (req.range.offset + done));
                if (n < 0) {
                    if (errno == EINTR) {
                        continue;
                    }
                    break;
                }
                if (n == 0) {
                    break;
                }
                done += (uint64_t) n;
            }

            {
                std::lock_guard<std::mutex> lock(mu_);
                in_flight_--;
                inflight_bytes_ -= std::min(inflight_bytes_, req.range.length);
                if (done == req.range.length) {
                    stats_.completed++;
                    stats_.bytes += done;
                } else {
                    stats_.errors++;
                }
                if (queue_.empty() && in_flight_ == 0) {
                    cv_idle_.notify_all();
                }
            }
        }
    }

    StreamerConfig            cfg_;
    int                       fd_;
    mutable std::mutex        mu_;
    std::condition_variable   cv_;
    std::condition_variable   cv_idle_;
    std::deque<ReadRequest>   queue_;
    std::vector<std::thread>  workers_;
    uint64_t                  inflight_bytes_ = 0;
    uint32_t                  in_flight_     = 0;
    bool                      stop_          = false;
    StreamStats               stats_;
};

// Dispatches each request to the right engine.
class CompositeReadScheduler final : public IReadScheduler {
public:
    CompositeReadScheduler(const ReadSchedulerConfig & cfg, int fd,
                           std::unique_ptr<IPageCacheWarmer> warmer,
                           std::unique_ptr<IStreamReader>  streamer)
        : cfg_(cfg), fd_(fd), warmer_(std::move(warmer)), streamer_(std::move(streamer)) {}

    const char * name() const override {
        return streamer_ ? "composite(warm+stream)" : "composite(warm)";
    }

    Status submit(const ReadRequest & req) override {
        if (req.policy == ReadPolicy::kCacheWarm || !streamer_) {
            return warmer_->submit(req.range);
        }
        return streamer_->submit(req);
    }

    Status wait_idle(uint32_t timeout_ms) override {
        Status st = warmer_->wait_idle(timeout_ms);
        if (streamer_) {
            const Status st2 = streamer_->wait_idle(timeout_ms);
            if (!st2.ok()) {
                return st2;
            }
        }
        return st;
    }

    WarmStats warm_stats() const override { return warmer_->stats(); }
    StreamStats stream_stats() const override {
        return streamer_ ? streamer_->stats() : StreamStats{};
    }
    uint64_t bytes_inflight() const override {
        return warmer_->queued_bytes() + (streamer_ ? streamer_->bytes_inflight() : 0);
    }

private:
    ReadSchedulerConfig             cfg_;
    int                             fd_;
    std::unique_ptr<IPageCacheWarmer> warmer_;
    std::unique_ptr<IStreamReader>  streamer_;
};

} // namespace

std::unique_ptr<IStreamReader> make_stream_reader(const StreamerConfig & cfg, int fd) {
    auto r = std::make_unique<PoolStreamReader>(cfg, fd);
    const Status st = r->init(cfg);
    if (!st.ok()) {
        return nullptr;
    }
    return r;
}

std::unique_ptr<IReadScheduler> make_read_scheduler(const ReadSchedulerConfig & cfg, int fd) {
    auto warmer = make_page_cache_warmer(cfg.warmer, fd);
    std::unique_ptr<IStreamReader> streamer;
    if (cfg.enable_streaming) {
        streamer = make_stream_reader(cfg.streamer, fd);
        if (!streamer) {
            AMP_WARN("amp: streaming reader unavailable, cache-warm only");
        }
    }
    return std::make_unique<CompositeReadScheduler>(cfg, fd, std::move(warmer), std::move(streamer));
}

} // namespace amp
