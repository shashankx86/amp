// Streaming reads: fetch bytes into a caller-owned buffer instead of the page cache.
//
// Two residency policies exist because they solve different problems:
//   kCacheWarm - the data should stay resident (it will be reused every ubatch)
//   kStream    - the data is used once and must NOT evict the resident set, so it is
//                read into a small reusable staging buffer
// The second policy is what lets amp keep a pinned hot set while still covering the
// cold tail, instead of LRU thrashing the whole working set.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "amp/io/warmer.h"
#include "amp/status.h"

namespace amp {

enum class ReadPolicy {
    kCacheWarm,  // bring into page cache and keep it there
    kStream,     // read into a caller buffer, do not pollute the page cache
};

const char * to_string(ReadPolicy p);

struct ReadRequest {
    ReadRange range;
    ReadPolicy policy  = ReadPolicy::kCacheWarm;
    void *    dst      = nullptr;  // required for kStream
    uint32_t  stream   = 0;       // scheduling hint (0 = default)
    uint64_t  deadline_us = 0;    // 0 = no deadline; used for lookahead depth control
};

struct StreamStats {
    uint64_t submitted  = 0;
    uint64_t completed  = 0;
    uint64_t bytes      = 0;
    uint64_t errors     = 0;
    double   busy_seconds = 0.0;

    std::string to_string() const;
};

struct StreamerConfig {
    uint32_t threads     = 2;
    uint32_t queue_depth = 64;
    uint64_t max_inflight_bytes = 256ull * 1024 * 1024;  // backpressure guard
};

// Unified read front-end. Internally dispatches to the page-cache warmer or the
// stream reader depending on the request policy.
class IReadScheduler {
public:
    virtual ~IReadScheduler() = default;

    virtual const char * name() const = 0;
    virtual Status      submit(const ReadRequest & req) = 0;
    virtual Status      wait_idle(uint32_t timeout_ms) = 0;

    virtual WarmStats   warm_stats() const = 0;
    virtual StreamStats stream_stats() const = 0;
    virtual uint64_t    bytes_inflight() const = 0;
};

struct ReadSchedulerConfig {
    WarmerConfig    warmer;
    StreamerConfig  streamer;
    bool            enable_streaming = true;
};

std::unique_ptr<IReadScheduler> make_read_scheduler(const ReadSchedulerConfig & cfg, int fd);

// Concrete stream reader, exposed for tools/tests that want raw pread throughput.
class IStreamReader {
public:
    virtual ~IStreamReader() = default;
    virtual const char * name() const = 0;
    virtual Status      init(const StreamerConfig & cfg) = 0;
    virtual Status      submit(const ReadRequest & req) = 0;
    virtual Status      wait_idle(uint32_t timeout_ms) = 0;
    virtual StreamStats stats() const = 0;
    virtual uint64_t    bytes_inflight() const = 0;
};

std::unique_ptr<IStreamReader> make_stream_reader(const StreamerConfig & cfg, int fd);

} // namespace amp
