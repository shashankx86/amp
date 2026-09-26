// Page-cache warming.
//
// This is amp's central trick. The model is a 12.19 GiB working set read cyclically; the
// page cache is ~11.5 GiB. Random 4 KiB faults stream from NVMe at 555 MB/s, while the
// same bytes read sequentially hit ~2.0 GB/s. So residency is not left to chance:
// amp explicitly warms the ranges it needs, in large sequential chunks, ahead of use.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "amp/status.h"

namespace amp {

// A byte range inside the model file.
struct ReadRange {
    uint64_t offset = 0;
    uint64_t length = 0;

    uint64_t end() const { return offset + length; }
    bool     empty() const { return length == 0; }
    bool     contains(uint64_t off) const { return off >= offset && off < end(); }
};

inline bool operator==(const ReadRange & a, const ReadRange & b) {
    return a.offset == b.offset && a.length == b.length;
}

struct WarmStats {
    uint64_t ranges_queued   = 0;
    uint64_t bytes_queued    = 0;
    uint64_t bytes_issued    = 0;
    uint64_t syscalls        = 0;
    uint64_t errors          = 0;

    std::string to_string() const;
};

struct WarmerConfig {
    uint64_t chunk_bytes  = 2ull * 1024 * 1024;  // per readahead() call
    uint32_t queue_depth  = 512;                 // bounded: never flood the queue
    uint32_t threads      = 2;                   // readahead() itself is async, so few threads
    bool     use_readahead_syscall = true;       // fall back to fadvise if unavailable
};

// Asynchronous page-cache warmer. warm() never blocks on I/O; wait_idle() does.
class IPageCacheWarmer {
public:
    virtual ~IPageCacheWarmer() = default;

    virtual const char * name() const = 0;
    virtual Status      submit(const ReadRange & range) = 0;
    virtual Status      wait_idle(uint32_t timeout_ms) = 0;
    virtual WarmStats   stats() const = 0;
    virtual uint64_t    queued_bytes() const = 0;
};

// Synchronous warm with progress reporting. Used at server start-up, where we *want*
// to block: the point is to convert a cold, random-fault working set into a resident one.
struct WarmProgress {
    uint64_t bytes_done     = 0;
    uint64_t bytes_total    = 0;
    double   seconds       = 0.0;
    double   bytes_per_sec  = 0.0;
    bool     finished       = false;
    std::function<void(const WarmProgress &)> on_progress;
};

// Reads [range) into the page cache with large sequential pread()s, discarding the data.
// Returns the achieved bandwidth, which is the number we care about.
Status warm_range_blocking(int fd, const ReadRange & range, uint64_t chunk_bytes,
                           WarmProgress * progress = nullptr);

Status warm_ranges_blocking(int fd, const std::vector<ReadRange> & ranges, uint64_t chunk_bytes,
                            WarmProgress * progress = nullptr);

// Factory: returns the best available asynchronous warmer.
std::unique_ptr<IPageCacheWarmer> make_page_cache_warmer(const WarmerConfig & cfg, int fd);

} // namespace amp
