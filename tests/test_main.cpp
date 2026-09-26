#include "test_harness.h"

#include "amp/bytes.h"
#include "amp/timing.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace amp_test {

std::vector<Case> & registry() {
    static std::vector<Case> r;
    return r;
}

void fail(const char * file, int line, const std::string & msg) {
    const char * base = strrchr(file, '/');
    fprintf(stderr, "  FAIL %s:%d: %s\n", base ? base + 1 : file, line, msg.c_str());
    fflush(stderr);
    abort();  // a failing assertion invalidates everything after it
}

int run_all(const char * suite_name) {
    int failed = 0;
    printf("[%s] running %zu cases\n", suite_name, registry().size());
    for (auto & c : registry()) {
        const int before = failed;
        printf("  %-40s", c.name.c_str());
        fflush(stdout);
        c.fn();
        printf("%s\n", failed == before ? "ok" : "FAILED");
    }
    printf("[%s] %s\n", suite_name, failed == 0 ? "ALL PASS" : "FAILURES");
    return failed;
}

} // namespace amp_test

int main(int argc, char ** argv) {
    const char * suite = argc > 1 ? argv[1] : "amp";
    return amp_test::run_all(suite) == 0 ? 0 : 1;
}

// ---------------------------------------------------------------- test_util

using namespace amp;

AMP_TEST(util_format_substitutes_strings) {
    AMP_CHECK_EQ(format("hello %s, you are %d", "bob", 42), std::string("hello bob, you are 42"));
    AMP_CHECK_EQ(format("plain"), std::string("plain"));
    AMP_CHECK_EQ(format("%s%%", "x"), std::string("x%"));
}

AMP_TEST(util_format_handles_floats) {
    AMP_CHECK_EQ(format("%.2f", 3.14159), std::string("3.14"));
    AMP_CHECK_EQ(format("%.3f", 0.5), std::string("0.500"));
}

AMP_TEST(util_human_bytes) {
    AMP_CHECK_EQ(human_bytes(512), std::string("512 B"));
    AMP_CHECK(human_bytes(kGiB) == std::string("1.00 GiB"));
    AMP_CHECK(human_bytes(1536ull * kMiB) == std::string("1.50 GiB"));
}

AMP_TEST(util_parse_bytes) {
    uint64_t v = 0;
    AMP_CHECK(parse_bytes("6141MiB", &v));
    AMP_CHECK_EQ(v, 6141ull * kMiB);
    AMP_CHECK(parse_bytes("4k", &v));
    AMP_CHECK_EQ(v, 4096ull);
    AMP_CHECK(parse_bytes("1.5GiB", &v));
    AMP_CHECK_EQ(v, (uint64_t) (1.5 * kGiB));
    AMP_CHECK(!parse_bytes("banana", &v));
    AMP_CHECK(parse_bytes("2048", &v));
    AMP_CHECK_EQ(v, 2048ull);
}

AMP_TEST(util_align) {
    AMP_CHECK_EQ(align_up(1000, 512), 1024ull);
    AMP_CHECK_EQ(align_down(1000, 512), 512ull);
    AMP_CHECK_EQ(align_up(512, 512), 512ull);
}

AMP_TEST(util_meminfo_readable) {
    const MemInfo mi = read_meminfo();
    AMP_CHECK(mi.mem_total_bytes > 8ull * kGiB);   // this box has 14+ GiB
    AMP_CHECK(mi.mem_available_bytes > 0);
    AMP_CHECK(mi.cached_bytes > 0);
}

AMP_TEST(util_stopwatch_monotonic) {
    Stopwatch sw;
    volatile double x = 0;
    for (int i = 0; i < 1000000; i++) {
        x += i * 0.5;
    }
    (void) x;
    AMP_CHECK(sw.elapsed_us() > 0.0);
}
