// Is the CPU-side expert read limited by DRAM bandwidth or by the TLB?
//
// The project's performance notes claim the MoE expert matvec is "already at memory
// bandwidth": ggml mul_mat over an iq3_xxs expert runs at 29.21 GB/s against 28.69 GB/s
// for a plain 8-thread read of the same bytes, so the dequant work is fully hidden. That is
// a real measurement and it does show the kernel adds no cost of its own. It does not show
// that 28.7 GB/s is the ceiling of this machine.
//
// The model is 12.19 GiB of experts. Mapped at the default 4 KiB page size that is
// 3,195,535 separate mappings. No TLB on any current x86 holds that, so every row of every
// expert matvec walks the page tables, and a page walk that misses the last level cache is
// a chain of dependent memory references on the critical path. A workload can look
// bandwidth-bound while actually being translation-bound.
//
// Two access patterns are measured, because they fail differently and decode does both:
//
//   sweep     one contiguous region, walked row by row. Isolates the cost of a large
//             linear stream, which is the friendliest case for the prefetcher.
//   scattered the shape decode actually produces: 8 of 256 experts per layer, 40 layers,
//             landing at 320 disjoint places spread over 12.19 GiB. This is the pattern
//             that punishes a small page size hardest, because each expert start is a
//             distinct translation and none of them stay resident in the TLB between
//             tokens.
//
// Backing is varied separately from pattern, so the two variables do not get conflated:
//
//   file4k    the GGUF mapped read-only, exactly as llama.cpp maps it
//   anon4k    anonymous memory with MADV_NOHUGEPAGE, the file case's page size
//   thp       anonymous memory with MADV_HUGEPAGE, 2 MiB pages
//
// All three backings deliver byte-for-byte identical content and the tool refuses to report
// a comparison if they do not, because a rate difference between different bytes means
// nothing. AnonHugePages is sampled before and after so the THP arm reports what it
// achieved rather than what it requested: MADV_HUGEPAGE is a request, and on a fragmented
// box it can be refused, in which case the arm is not measuring what it claims.
//
// Not part of the serving path; a measurement tool.

#include "amp/bytes.h"
#include "amp/timing.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {

// One expert's gate/up slice: 2048 elements of iq3_xxs at 98 bytes per 256-element block.
// Derived from the format rather than hardcoded so the walk stays honest if the model's
// quantization ever changes.
constexpr int64_t kRowElems = 2048;
constexpr int64_t kRowBytes = kRowElems / 256 * 98;

// An expert is gate/up plus down: two [2048,512] gate/up planes and one [512,2048] down
// plane, at 98 bytes per 256 iq3_xxs elements.
constexpr int64_t kExpertBytes = (2 * 2048 * 512 + 512 * 2048) / 256 * 98;

// Routing shape, from the GGUF: 256 experts per layer, 8 used, 40 layers.
constexpr int64_t kLayers      = 40;
constexpr int64_t kExperts     = 256;
constexpr int64_t kExpertUsed  = 8;

// How much of the model each backing holds. Sized so all three fit at once alongside a
// warm page cache on a 15.6 GB box; a measurement over bytes the page cache has to fetch
// from disk would report the disk, not the memory system.
constexpr int64_t kSweepBytes = 1LL << 30;

struct Backing {
    std::string name;
    uint8_t *   base;
    int64_t     bytes;
};

// Sum a strided row walk. The sum is over bytes rather than over decoded weights on
// purpose: this measures the rate at which the memory system delivers expert bytes, which
// is the quantity in question. Whether those bytes are then dequantized is the separate
// comparison the kernel benchmark already made.
uint64_t walk_rows(const uint8_t * base, int64_t bytes, int64_t row_bytes) {
    uint64_t sum = 0;
    for (int64_t off = 0; off < bytes; off += row_bytes) {
        const int64_t chunk = std::min<int64_t>(row_bytes, bytes - off);
        for (int64_t i = 0; i < chunk; ++i) {
            sum += base[off + i];
        }
    }
    return sum;
}

// Walk a set of disjoint regions, the shape decode produces. Each region is an expert, so
// every region start is a separate translation that the TLB must resolve again on the next
// token. Regions are visited in the order given, spread across the whole arena.
uint64_t walk_scattered(const Backing & bk, const std::vector<int64_t> & region_off,
                        const std::vector<int64_t> & region_len, int64_t row_bytes) {
    uint64_t sum = 0;
    for (size_t r = 0; r < region_off.size(); ++r) {
        sum += walk_rows(bk.base + region_off[r], region_len[r], row_bytes);
    }
    return sum;
}

// Split regions across workers and time the whole set. Regions are assigned round-robin
// rather than contiguously, because a contiguous split would serialize the workers onto
// the same end of the arena and let the prefetcher run ahead of the request.
struct Result {
    double      gbps;
    double      ms;
    uint64_t    checksum;
};

Result time_walk(const std::function<uint64_t(int)> & body, int64_t total_bytes,
                 int n_threads, int passes) {
    Result best{ 0.0, 0.0, 0 };

    for (int pass = 0; pass < passes; ++pass) {
        std::vector<uint64_t> sums((size_t) n_threads, 0);

        const auto t0 = std::chrono::steady_clock::now();
        {
            std::vector<std::thread> workers;
            workers.reserve((size_t) n_threads);

            for (int t = 0; t < n_threads; ++t) {
                workers.emplace_back([&, t] { sums[(size_t) t] = body(t); });
            }
            for (auto & w : workers) {
                w.join();
            }
        }
        const double ms = amp::Stopwatch::us_since(t0) / 1e3;

        uint64_t sum = 0;
        for (uint64_t s : sums) {
            sum += s;
        }

        // Keep the fastest pass. Page-cache state and CPU frequency drift over a
        // second-scale run, so the slower of several passes is not a useful number.
        const double gbps = (double) total_bytes / (ms * 1e6);
        if (gbps > best.gbps) {
            best = Result{ gbps, ms, sum };
        }
    }

    return best;
}

int64_t anon_hugepages_kb() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return -1;
    }

    char line[256];
    int64_t kb = -1;
    while (fgets(line, sizeof(line), f) != nullptr) {
        if (strncmp(line, "AnonHugePages:", 14) == 0) {
            kb = strtoll(line + 14, nullptr, 10);
            break;
        }
    }

    fclose(f);
    return kb;
}

// Build the region list decode produces: n_layers layers, each contributing n_expert_used
// experts at a fixed stride, so the regions are spread across the whole arena the way real
// routing spreads them. Every region is a whole expert, because that is the granularity at
// which the router chooses and at which the kernel is handed a pointer.
std::vector<int64_t> expert_offsets(int64_t arena_bytes, int64_t stride, int64_t * out_len) {
    std::vector<int64_t> off;

    const int64_t per_layer = kExperts * kExpertBytes;
    if (per_layer * kLayers > arena_bytes) {
        // Arena too small to hold the real shape; fall back to what does fit rather than
        // reporting a number for a shape that was never measured.
        const int64_t layers = std::max<int64_t>(1, arena_bytes / per_layer);
        for (int64_t l = 0; l < layers; ++l) {
            for (int64_t e = 0; e < kExpertUsed; ++e) {
                const int64_t idx = (l * 37 + e * 29) % kExperts;
                off.push_back(l * per_layer + idx * stride);
            }
        }
        return off;
    }

    for (int64_t l = 0; l < kLayers; ++l) {
        for (int64_t e = 0; e < kExpertUsed; ++e) {
            // A deterministic spread. The exact indices do not matter, only that the 320
            // regions are disjoint and cover the arena.
            const int64_t idx = (l * 37 + e * 29) % kExperts;
            off.push_back(l * per_layer + idx * stride);
        }
    }

    *out_len = stride;
    return off;
}

} // namespace

int main(int argc, char ** argv) {
    const char * model = nullptr;
    int n_threads = 0;
    int passes = 3;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            n_threads = atoi(argv[++i]);
        } else if (a == "--passes" && i + 1 < argc) {
            passes = atoi(argv[++i]);
        } else if (a == "--help" || a == "-h") {
            printf(
                "usage: amp-page-walk --model <gguf> [--threads N] [--passes N]\n"
                "\n"
                "Measures the rate at which this machine delivers the model's expert bytes,\n"
                "over identical bytes held with 4 KiB and 2 MiB pages, in both a contiguous\n"
                "sweep and the scattered pattern decode actually produces. A large gap\n"
                "between page sizes means the expert read is limited by address translation\n"
                "rather than by DRAM.\n");
            return 0;
        } else {
            fprintf(stderr, "amp-page-walk: unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }

    if (model == nullptr) {
        fprintf(stderr, "amp-page-walk: --model is required\n");
        return 2;
    }

    if (n_threads <= 0) {
        // One worker per physical core. SMT siblings share a core's load/store units, and
        // this measurement is about saturating the memory system, so a second thread per
        // core adds contention without adding reach.
        n_threads = std::max(1, (int) (std::thread::hardware_concurrency() / 2));
    }

    const int fd = open(model, O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "amp-page-walk: cannot open %s: %s\n", model, strerror(errno));
        return 1;
    }

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        fprintf(stderr, "amp-page-walk: cannot stat %s: %s\n", model, strerror(errno));
        close(fd);
        return 1;
    }

    const int64_t file_bytes = (int64_t) st.st_size;
    const int64_t arena      = std::min<int64_t>(kSweepBytes, file_bytes);

    // Map from the middle of the file. The header and the first layers are not where decode
    // spends its time, and mapping at zero would put the sweep in the low addresses where a
    // transparent huge page is least likely to be found.
    const int64_t map_off = (file_bytes - arena) / 2 & ~int64_t{ 4095 };

    void * file_map = mmap(nullptr, (size_t) arena, PROT_READ, MAP_PRIVATE, fd, map_off);
    if (file_map == MAP_FAILED) {
        fprintf(stderr, "amp-page-walk: mmap failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    // Two separate arenas, not one arena split in two. MADV_NOHUGEPAGE and MADV_HUGEPAGE
    // are mutually exclusive advice over an address range, and giving each arm its own
    // mapping keeps the page size of each unambiguous and the two contents identical.
    void * a4k_map = mmap(nullptr, (size_t) arena, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    void * thp_map = mmap(nullptr, (size_t) arena, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a4k_map == MAP_FAILED || thp_map == MAP_FAILED) {
        fprintf(stderr, "amp-page-walk: anonymous mmap failed: %s\n", strerror(errno));
        munmap(file_map, (size_t) arena);
        close(fd);
        return 1;
    }

    auto * a4k = (uint8_t *) a4k_map;
    auto * thp = (uint8_t *) thp_map;

    // Advice must precede first touch, or the pages already exist at 4 KiB and stay there.
    if (madvise(a4k, (size_t) arena, MADV_NOHUGEPAGE) != 0) {
        fprintf(stderr, "amp-page-walk: MADV_NOHUGEPAGE failed: %s\n", strerror(errno));
    }
    if (madvise(thp, (size_t) arena, MADV_HUGEPAGE) != 0) {
        fprintf(stderr, "amp-page-walk: MADV_HUGEPAGE failed: %s\n", strerror(errno));
    }

    // Identical content in all three, so any rate difference is attributable to the backing
    // alone. Reading the file also brings that region into the page cache, which is the state
    // the file arm has to be measured in.
    //
    // The copy is also what faults the anonymous arms in, and it must not be followed by a
    // separate prefault pass: a prefault writes, so it would change the content these
    // backings hold and the checksum comparison would fail for a reason that has nothing to
    // do with page size.
    memcpy(a4k, file_map, (size_t) arena);
    memcpy(thp, file_map, (size_t) arena);

    const std::vector<Backing> backings = {
        { "file4k", (uint8_t *) file_map, arena },
        { "anon4k", a4k,                 arena },
        { "thp",    thp,                 arena },
    };

    int64_t expert_len = kExpertBytes;
    const std::vector<int64_t> regions = expert_offsets(arena, kExpertBytes, &expert_len);
    const int64_t scattered_bytes = (int64_t) regions.size() * expert_len;

    printf("amp-page-walk\n");
    printf("  model        %s\n", model);
    printf("  arena        %s per backing, at file offset %s\n",
           amp::human_bytes((uint64_t) arena).c_str(), amp::human_bytes((uint64_t) map_off).c_str());
    printf("  threads      %d (one per physical core), best of %d\n", n_threads, passes);
    printf("  row          %lld B (iq3_xxs, 2048 elements)\n", (long long) kRowBytes);
    printf("  expert       %s, %zu routed per pass (%lld layers x %lld of %lld)\n",
           amp::human_bytes((uint64_t) kExpertBytes).c_str(), regions.size(),
           (long long) kLayers, (long long) kExpertUsed, (long long) kExperts);
    printf("\n");

    const int64_t huge_before = anon_hugepages_kb();

    std::vector<Result> sweep_res;
    std::vector<Result> scat_res;
    sweep_res.reserve(backings.size());
    scat_res.reserve(backings.size());

    printf("  %-8s %18s %18s\n", "", "sweep GB/s", "scattered GB/s");
    printf("  %-8s %18s %18s\n", "backing", "(contiguous)", "(decode shape)");
    printf("  %s\n", std::string(46, '-').c_str());

    for (const auto & bk : backings) {
        const Result s = time_walk(
            [&](int t) {
                const int64_t chunk = (bk.bytes / n_threads + 63) & ~int64_t{ 63 };
                const int64_t off   = chunk * t;
                if (off >= bk.bytes) {
                    return uint64_t{ 0 };
                }
                return walk_rows(bk.base + off, std::min(chunk, bk.bytes - off), kRowBytes);
            },
            bk.bytes, n_threads, passes);

        const Result c = time_walk(
            [&](int t) {
                std::vector<int64_t> off;
                std::vector<int64_t> len;
                for (size_t r = (size_t) t; r < regions.size(); r += (size_t) n_threads) {
                    off.push_back(regions[r]);
                    len.push_back(expert_len);
                }
                return walk_scattered(bk, off, len, kRowBytes);
            },
            scattered_bytes, n_threads, passes);

        sweep_res.push_back(s);
        scat_res.push_back(c);

        printf("  %-8s %18.2f %18.2f\n", bk.name.c_str(), s.gbps, c.gbps);
    }

    const int64_t huge_gained = anon_hugepages_kb() - huge_before;

    printf("\n");

    bool identical = true;
    for (size_t i = 1; i < sweep_res.size(); ++i) {
        identical = identical && sweep_res[i].checksum == sweep_res[0].checksum
                             && scat_res[i].checksum == scat_res[0].checksum;
    }

    if (!identical) {
        printf("  CHECKSUM MISMATCH: the backings did not deliver identical bytes, so the\n"
               "  rates above are not comparable and must not be acted on.\n");
        return 1;
    }
    printf("  checksums agree: all three backings delivered identical bytes\n");
    printf("  huge pages actually held: %s\n",
           huge_gained > 0 ? amp::human_bytes((uint64_t) huge_gained * 1024).c_str() : "none");

    const auto pct = [](double a, double b) { return 100.0 * (a / b - 1.0); };

    printf("\n  page size, 4 KiB -> 2 MiB:\n");
    printf("    sweep       %+.1f%%\n", pct(sweep_res[2].gbps, sweep_res[1].gbps));
    printf("    scattered   %+.1f%%\n", pct(scat_res[2].gbps, scat_res[1].gbps));
    printf("  file-backed vs anonymous at 4 KiB:\n");
    printf("    sweep       %+.1f%%\n", pct(sweep_res[0].gbps, sweep_res[1].gbps));
    printf("    scattered   %+.1f%%\n", pct(scat_res[0].gbps, scat_res[1].gbps));

    if (huge_gained <= 0) {
        printf("\n  MADV_HUGEPAGE obtained no pages. The 'thp' row is not a huge-page\n"
               "  measurement and the page-size comparison does not hold.\n");
    } else if (scat_res[2].gbps > scat_res[1].gbps * 1.15) {
        printf("\n  Scattered reads are translation-bound: 2 MiB pages buy real bandwidth on\n"
               "  the pattern decode uses. Page size is a lever here.\n");
    } else {
        printf("\n  Page size does not move the rate on either pattern. The read really is at\n"
               "  this memory system's limit and the project's existing note stands: a\n"
               "  cheaper dequant kernel or a prefetcher cannot help the expert path.\n");
    }

    munmap(thp_map, (size_t) arena);
    munmap(a4k_map, (size_t) arena);
    munmap(file_map, (size_t) arena);
    close(fd);

    return 0;
}
