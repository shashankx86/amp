// Why the M3c prefetch is a tie: is the expert matvec limited by memory or by arithmetic?
//
// The prefetch measurements came out flat. "Prefetching does not help" has two very
// different explanations:
//
//   memory-limited     -> the bytes are already there, nothing to fetch ahead of time
//   arithmetic-limited -> the dequant and dot-product work is the wall, and bytes arriving
//                        earlier cannot change how long that arithmetic takes
//
// Both predict a tie, so the tie alone does not say which. This separates them. Over the same
// bytes, one expert's slice of one layer's gate matrix, it measures:
//
//   plain  : sum the bytes with 8 threads. The memory system's rate for this data.
//   kernel : the real ggml mul_mat over the same bytes at batch 1. What decode actually does.
//
// If the kernel rate is near the plain rate, the kernel is close to memory-limited, so a better
// memory schedule could in principle matter. If it is far below, the kernel is
// arithmetic-limited, the memory system is idle waiting to be asked, and no prefetch depth can
// close the gap. That distinction decides between "write a prefetcher" and "write a cheaper
// dequant kernel".
//
// Addresses come from the GGUF header's own tensor offsets. That arithmetic is validated
// against the file size below and aborts rather than reporting a number it cannot justify.
//
// Not part of the serving path; a measurement tool.

#include "ggml.h"
#include "ggml-cpu.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cinttypes>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <thread>
#include <vector>

static double now_s() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

static int32_t gguf_type_to_ggml(int32_t t) {
    switch (t) {
        case  0: return GGML_TYPE_F32;
        case  1: return GGML_TYPE_F16;
        case  2: return GGML_TYPE_Q4_0;
        case  3: return GGML_TYPE_Q4_1;
        case  6: return GGML_TYPE_Q5_0;
        case  7: return GGML_TYPE_Q5_1;
        case  8: return GGML_TYPE_Q8_0;
        case 10: return GGML_TYPE_Q2_K;
        case 11: return GGML_TYPE_Q3_K;
        case 12: return GGML_TYPE_Q4_K;
        case 13: return GGML_TYPE_Q5_K;
        case 14: return GGML_TYPE_Q6_K;
        case 15: return GGML_TYPE_Q8_K;
        case 16: return GGML_TYPE_IQ2_XXS;
        case 17: return GGML_TYPE_IQ2_XS;
        case 18: return GGML_TYPE_IQ3_XXS;
        case 19: return GGML_TYPE_IQ1_S;
        case 20: return GGML_TYPE_IQ4_NL;
        case 21: return GGML_TYPE_IQ3_S;
        case 22: return GGML_TYPE_IQ2_S;
        case 23: return GGML_TYPE_IQ4_XS;
        case 29: return GGML_TYPE_IQ1_M;
        case 30: return GGML_TYPE_BF16;
        default: return GGML_TYPE_COUNT;
    }
}

struct tensor_info {
    std::string name;
    std::vector<uint64_t> dims;
    ggml_type type;
    uint64_t offset;
    uint64_t nbytes;
};

static void die(const char * msg) { fprintf(stderr, "kernel-bound: %s\n", msg); exit(1); }

// Sum one byte per 64 B line, so this streams like a real consumer rather than collapsing into
// a single cache line the way a naive byte loop would.
static double plain_rate(const char * p, size_t bytes, int nthreads, int reps, int warm) {
    volatile double sink = 0;
    std::vector<std::thread> th;
    double best = 0;

    for (int r = 0; r < reps + warm; r++) {
        double t0 = now_s();
        for (int t = 0; t < nthreads; t++) {
            th.emplace_back([&, t] {
                double s = 0;
                const size_t n = bytes/(size_t) nthreads;
                const char * q = p + n*(size_t) t;
                for (size_t i = 0; i < n; i += 64) s += *(const volatile char *)(q + i);
                sink = s;
            });
        }
        for (auto & x : th) x.join();
        th.clear();
        const double gbs = bytes/(now_s() - t0)/1e9;
        if (r >= warm && gbs > best) best = gbs;
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s MODEL.gguf [LAYER] [NTHREADS] [REPS]\n", argv[0]); return 2; }
    const char * model = argv[1];
    const int layer    = argc > 2 ? atoi(argv[2]) : 20;
    const int nthreads = argc > 3 ? atoi(argv[3]) : 8;
    const int reps     = argc > 4 ? atoi(argv[4]) : 20;

    FILE * f = fopen(model, "rb");
    if (!f) die("cannot open model");

    auto pod = [&](auto & out) { if (fread(&out, 1, sizeof(out), f) != sizeof(out)) die("short read"); };
    auto rd_str = [&]() -> std::string {
        uint64_t n; pod(n);
        std::string s(n, '\0');
        if (n && fread(&s[0], 1, n, f) != n) die("short string");
        return s;
    };
    static const size_t vsz[] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    std::function<void(int32_t, uint64_t)> skip_val;
    skip_val = [&](int32_t t, uint64_t n) {
        if (t == 8) { for (uint64_t i = 0; i < n; i++) rd_str(); }
        else if (t == 9) { int32_t et; pod(et); uint64_t en; pod(en); skip_val(et, en*n); }
        else if (fseek(f, (long) (vsz[t]*n), SEEK_CUR) != 0) die("seek failed");
    };
    auto read_scalar = [&](int32_t t) -> uint64_t {
        if (t == 8) return 0;
        if (t == 9) die("unexpected array where scalar expected");
        uint64_t v = 0;
        if (fread(&v, 1, vsz[t], f) != vsz[t]) die("short scalar");
        return v;
    };

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "GGUF", 4) != 0) die("not a GGUF file");
    uint32_t ver; pod(ver);
    uint64_t n_tensors, n_kv; pod(n_tensors); pod(n_kv);

    uint64_t alignment = 32;
    for (uint64_t i = 0; i < n_kv; i++) {
        const std::string k = rd_str();
        int32_t t; pod(t);
        if (k == "general.alignment") alignment = read_scalar(t);
        else skip_val(t, 1);
    }

    std::vector<tensor_info> tis;
    for (uint64_t i = 0; i < n_tensors; i++) {
        tensor_info ti;
        ti.name = rd_str();
        uint32_t nd; pod(nd);
        for (uint32_t d = 0; d < nd; d++) { uint64_t v; pod(v); ti.dims.push_back(v); }
        int32_t raw; pod(raw);
        ti.type = (ggml_type) gguf_type_to_ggml(raw);
        if (ti.type == GGML_TYPE_COUNT) die("unhandled tensor type in header");
        pod(ti.offset);
        ti.nbytes = ggml_row_size(ti.type, (int64_t) ti.dims[0]);
        for (size_t d = 1; d < ti.dims.size(); d++) ti.nbytes *= ti.dims[d];
        tis.push_back(ti);
    }

    // The data section begins at the next `alignment` boundary after the header.
    const long header_end = ftell(f);
    const uint64_t data_start = (uint64_t) ((header_end + (long) alignment - 1) & ~((long) alignment - 1));

    uint64_t span = 0;
    for (const auto & ti : tis) span = std::max(span, ti.offset + ti.nbytes);
    fclose(f);

    int fd = open(model, O_RDONLY);
    if (fd < 0) die("cannot open model for mmap");
    struct stat st;
    if (fstat(fd, &st) != 0) die("fstat failed");

    if (data_start + span > (uint64_t) st.st_size) {
        fprintf(stderr, "kernel-bound: computed data span %" PRIu64 " + %" PRIu64
                        " exceeds file size %" PRIu64 "; header arithmetic is wrong\n",
                data_start, span, (uint64_t) st.st_size);
        return 1;
    }

    const char * base = (const char *) mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) die("mmap failed");

    char name[128];
    snprintf(name, sizeof(name), "blk.%d.ffn_gate_exps.weight", layer);
    const tensor_info * target = nullptr;
    for (const auto & ti : tis) if (ti.name == name) target = &ti;
    if (!target) die("tensor not found in header");

    const int64_t ne00 = (int64_t) target->dims[0];
    const int64_t ne01 = (int64_t) target->dims[1];
    const int64_t ne02 = (int64_t) target->dims[2];

    // One expert is ne00 x ne01 elements. row_size takes an element count, so passing ne01
    // would size a single quantised row instead of the expert.
    const size_t expert_bytes = ggml_row_size(target->type, ne00)*(size_t) ne01;
    const char * expert = base + data_start + target->offset;

    // Decode streams 8 experts per layer across 40 layers, so its working set is far larger
    // than cache. Benchmarking one expert alone would sit in L2 and flatter the kernel, so the
    // timed region below sweeps the whole tensor, which is the same vec_dot over the same
    // bytes at a realistic working-set size.
    const size_t total_bytes = expert_bytes*(size_t) ne02;

    printf("model    : %s (gguf v%u, %" PRIu64 " tensors, alignment %" PRIu64 ")\n",
           model, ver, n_tensors, alignment);
    printf("tensor   : %s  type=%s\n", name, ggml_type_name((ggml_type) target->type));
    printf("geometry : ne0=%" PRId64 " ne1=%" PRId64 " ne2=%" PRId64 "\n", ne00, ne01, ne02);
    printf("bytes    : %.2f MiB per expert, %.2f MiB for all %" PRId64 " experts\n",
           expert_bytes/1048576.0, total_bytes/1048576.0, ne02);
    printf("threads  : %d, reps %d\n\n", nthreads, reps);

    const double plain = plain_rate(expert, total_bytes, nthreads, reps, 2);
    printf("plain read, whole tensor : %7.2f GB/s   (%.2f MiB in %.3f ms)\n",
           plain, total_bytes/1048576.0, 1000.0*total_bytes/(plain*1e9));

    // The real kernel: mul_mat of that expert against a single-token activation, which is
    // exactly the shape decode uses.
    ggml_init_params ip = { 64u*1024*1024, nullptr, /*.no_alloc=*/true };
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) die("ggml_init failed");

    ggml_backend_t be = ggml_backend_cpu_init();
    if (!be) die("cpu backend init failed");

    // Without this the backend picks its own thread count and the thread argument only affects
    // the plain read, which would make the two rates incomparable and the sweep meaningless.
    ggml_backend_cpu_set_n_threads(be, nthreads);

    ggml_tensor * w = ggml_new_tensor_2d(ctx, target->type, ne00, ne01*ne02);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ne00, 1);
    w->data = (void *) expert;                 // point at the mmap; never copied
    w->buffer = nullptr;

    ggml_tensor * out = ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) die("backend alloc failed");
    w->buffer = nullptr;                       // alloc may have claimed it; keep it on the mmap
    w->data   = (void *) expert;

    {
        std::vector<float> xv((size_t) ne00);
        for (int64_t i = 0; i < ne00; i++) xv[(size_t) i] = 0.01f*(float) ((i % 17) - 8);
        if (!x->data) die("activation has no data");
        memcpy(x->data, xv.data(), (size_t) ne00*sizeof(float));
    }

    for (int i = 0; i < 3; i++) ggml_backend_graph_compute(be, gf);

    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        const double t0 = now_s();
        ggml_backend_graph_compute(be, gf);
        const double dt = now_s() - t0;
        if (dt < best) best = dt;
    }

    const double kernel_rate = total_bytes/best/1e9;
    printf("ggml mul_mat, batch 1  : %7.2f GB/s   (%.2f MiB in %.3f ms)\n",
           kernel_rate, total_bytes/1048576.0, best*1000.0);
    printf("\n");
    printf("kernel as %% of plain    : %5.1f %%\n", 100.0*kernel_rate/plain);
    printf("\n");

    // If the kernel were purely memory-limited it would run at the plain read rate. The gap
    // between 1/ratio and 1 is the compute the kernel adds on top of moving the bytes, so the
    // run decomposes into a memory part and a dequant/dot part. Report the split rather than a
    // pass/fail, because any single threshold here would be arbitrary and would pick a
    // conclusion instead of measuring one.
    const double mem_frac = kernel_rate/plain;
    const double kernel_over_mem = 1.0/mem_frac;

    printf("  the kernel takes %.2fx as long as simply reading the same bytes\n", kernel_over_mem);
    if (kernel_over_mem <= 1.02) {
        printf("  so it moves those bytes as fast as a plain read of them, within noise. The\n");
        printf("  dequant work is hidden behind the memory traffic. There is no exposed latency\n");
        printf("  for a prefetcher to recover, because nothing is being waited on that a\n");
        printf("  prefetcher could have started earlier.\n");
    } else {
        printf("  so, of the kernel's time, roughly %.0f%% is moving the bytes and %.0f%% is the\n",
               100.0/kernel_over_mem, 100.0*(1.0 - 1.0/kernel_over_mem));
        printf("  dequant and dot work on top of it.\n");
    }
    printf("\n");
    printf("  A prefetcher can only ever address the memory part, and only the part of it that\n");
    printf("  is exposed latency rather than bytes the hardware prefetcher already streams.\n");
    printf("  Compare this kernel's rate against what decode actually achieves over the same\n");
    printf("  bytes: if decode is far below the isolated kernel, most of decode is not the\n");
    printf("  expert matvec at all, and no expert-side trick can reach it.\n");

    ggml_backend_buffer_free(buf);
    ggml_backend_free(be);
    ggml_free(ctx);
    munmap((void *) base, st.st_size);
    close(fd);
    return 0;
}
