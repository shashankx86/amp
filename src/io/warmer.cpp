#include "amp/io/warmer.h"

#include "amp/bytes.h"
#include "amp/format.h"
#include "amp/log.h"
#include "amp/timing.h"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

namespace amp {

std::string WarmStats::to_string() const {
    return format("ranges=%s queued=%s issued=%s syscalls=%s errors=%s",
                  human_count(ranges_queued).c_str(), human_bytes(bytes_queued).c_str(),
                  human_bytes(bytes_issued).c_str(), human_count(syscalls).c_str(),
                  human_count(errors).c_str());
}

namespace {

// Worker that turns queued ranges into readahead()/fadvise() calls.
class ReadaheadWarmer final : public IPageCacheWarmer {
public:
    ReadaheadWarmer(const WarmerConfig & cfg, int fd) : cfg_(cfg), fd_(fd) {
        nthreads_ = std::max(1u, cfg_.threads);
        workers_.reserve(nthreads_);
        for (uint32_t i = 0; i < nthreads_; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~ReadaheadWarmer() override {
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

    const char * name() const override { return cfg_.use_readahead_syscall ? "readahead" : "fadvise"; }

    Status submit(const ReadRange & range) override {
        if (range.empty()) {
            return Status::OK();
        }
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stop_) {
                return Status::Error("warmer stopped");
            }
            if (queue_.size() >= cfg_.queue_depth) {
                // Bounded queue: fall back to issuing inline rather than growing without limit.
                ++stats_.errors;
                return Status::Error("warm queue full");
            }
            queue_.push_back(range);
            stats_.ranges_queued++;
            stats_.bytes_queued += range.length;
        }
        cv_.notify_one();
        return Status::OK();
    }

    Status wait_idle(uint32_t timeout_ms) override {
        std::unique_lock<std::mutex> lock(mu_);
        const bool idle = cv_idle_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return queue_.empty() && in_flight_ == 0;
        });
        return idle ? Status::OK() : Status::Error("warmer wait_idle timed out");
    }

    WarmStats stats() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_;
    }

    uint64_t queued_bytes() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return stats_.bytes_queued - stats_.bytes_issued;
    }

private:
    void issue(const ReadRange & r) {
        // Split into chunk-sized pieces; readahead() is asynchronous, so a tight loop here
        // builds queue depth in the block layer instead of serialising.
        for (uint64_t off = 0; off < r.length; off += cfg_.chunk_bytes) {
            const uint64_t len = std::min(cfg_.chunk_bytes, r.length - off);
            if (cfg_.use_readahead_syscall) {
                // readahead() returns the number of bytes queued; -1 means the range is cached.
                const ssize_t n = ::readahead(fd_, (off_t) (r.offset + off), (size_t) len);
                if (n < 0) {
                    ++stats_.errors;
                }
            } else {
                posix_fadvise(fd_, (off_t) (r.offset + off), (off_t) len, POSIX_FADV_WILLNEED);
            }
            stats_.bytes_issued += len;
            stats_.syscalls++;
        }
    }

    void worker_loop() {
        for (;;) {
            ReadRange r;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) {
                    return;
                }
                r = queue_.front();
                queue_.pop_front();
                in_flight_++;
            }
            issue(r);
            {
                std::lock_guard<std::mutex> lock(mu_);
                in_flight_--;
                if (queue_.empty() && in_flight_ == 0) {
                    cv_idle_.notify_all();
                }
            }
        }
    }

    WarmerConfig                cfg_;
    int                         fd_;
    mutable std::mutex          mu_;
    std::condition_variable     cv_;
    std::condition_variable     cv_idle_;
    std::deque<ReadRange>       queue_;
    std::vector<std::thread>    workers_;
    uint32_t                    nthreads_ = 1;
    uint32_t                    in_flight_ = 0;
    bool                        stop_     = false;
    WarmStats                   stats_;
};

} // namespace

std::unique_ptr<IPageCacheWarmer> make_page_cache_warmer(const WarmerConfig & cfg, int fd) {
    auto w = std::make_unique<ReadaheadWarmer>(cfg, fd);
    AMP_INFO("amp: page-cache warmer = ", w->name(), " chunk=", human_bytes(cfg.chunk_bytes),
             " threads=", cfg.threads);
    return w;
}

Status warm_range_blocking(int fd, const ReadRange & range, uint64_t chunk_bytes,
                           WarmProgress * progress) {
    if (range.empty()) {
        return Status::OK();
    }
    if (chunk_bytes == 0) {
        chunk_bytes = 2ull * 1024 * 1024;
    }

    // One reusable buffer: allocating per chunk would churn the very page cache we
    // are trying to fill. 2 MiB is the sweet spot measured on this NVMe.
    const size_t buf_size = (size_t) std::min<uint64_t>(chunk_bytes, 8ull * 1024 * 1024);
    std::vector<uint8_t> buf(buf_size);

    Stopwatch sw;
    uint64_t done = 0;
    while (done < range.length) {
        const size_t len = (size_t) std::min<uint64_t>(buf_size, range.length - done);
        ssize_t     n   = -1;
        do {
            n = pread(fd, buf.data(), len, (off_t) (range.offset + done));
        } while (n < 0 && errno == EINTR);
        if (n <= 0) {
            return Status::Errorf("warm pread at +%llu returned %zd (%s)",
                                  (unsigned long long) done, n, strerror(errno));
        }
        done += (uint64_t) n;
        if (progress) {
            progress->bytes_done     = done;
            progress->bytes_total    = range.length;
            progress->seconds       = sw.elapsed_s();
            progress->bytes_per_sec  = sw.elapsed_s() > 0 ? (double) done / sw.elapsed_s() : 0.0;
            if (progress->on_progress) {
                progress->on_progress(*progress);
            }
        }
    }
    if (progress) {
        progress->finished = true;
    }
    return Status::OK();
}

Status warm_ranges_blocking(int fd, const std::vector<ReadRange> & ranges, uint64_t chunk_bytes,
                            WarmProgress * progress) {
    uint64_t total = 0;
    for (const auto & r : ranges) {
        total += r.length;
    }
    uint64_t done_all = 0;
    const Stopwatch sw;
    for (const auto & r : ranges) {
        WarmProgress sub;
        sub.bytes_total    = total;
        sub.on_progress    = [&](const WarmProgress & p) {
            if (progress) {
                progress->bytes_done    = done_all + p.bytes_done;
                progress->bytes_total   = total;
                progress->seconds       = sw.elapsed_s();
                progress->bytes_per_sec = sw.elapsed_s() > 0
                                              ? (double) (done_all + p.bytes_done) / sw.elapsed_s()
                                              : 0.0;
                if (progress->on_progress) {
                    progress->on_progress(*progress);
                }
            }
        };
        const Status st = warm_range_blocking(fd, r, chunk_bytes, &sub);
        if (!st.ok()) {
            return st;
        }
        done_all += r.length;
    }
    if (progress) {
        progress->bytes_done    = done_all;
        progress->bytes_total   = total;
        progress->seconds       = sw.elapsed_s();
        progress->bytes_per_sec = sw.elapsed_s() > 0 ? (double) done_all / sw.elapsed_s() : 0.0;
        progress->finished      = true;
    }
    return Status::OK();
}

} // namespace amp
