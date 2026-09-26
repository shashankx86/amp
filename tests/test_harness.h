// Test harness: no external dependency. Register cases with AMP_TEST(name) { ... }.
#pragma once

#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "amp/format.h"

namespace amp_test {

struct Case {
    std::string           name;
    std::function<void()> fn;
};

std::vector<Case> & registry();

struct Registrar {
    Registrar(const char * name, std::function<void()> fn) {
        registry().push_back({ name, std::move(fn) });
    }
};

void fail(const char * file, int line, const std::string & msg);

int run_all(const char * suite_name);

} // namespace amp_test

#define AMP_TEST(name)                                                        \
    static void name();                                                       \
    static ::amp_test::Registrar reg_##name(#name, name);                     \
    static void name()

#define AMP_CHECK(cond)                                                       \
    do {                                                                      \
        if (!(cond)) {                                                        \
            ::amp_test::fail(__FILE__, __LINE__, "check failed: " #cond);     \
        }                                                                     \
    } while (0)

#define AMP_CHECK_MSG(cond, msg)                                              \
    do {                                                                      \
        if (!(cond)) {                                                        \
            ::amp_test::fail(__FILE__, __LINE__,                              \
                             std::string("check failed: " #cond " -- ") + (msg)); \
        }                                                                     \
    } while (0)

#define AMP_CHECK_EQ(a, b)                                                    \
    do {                                                                      \
        const auto amp_a_ = (a);                                              \
        const auto amp_b_ = (b);                                              \
        if (!(amp_a_ == amp_b_)) {                                            \
            ::amp_test::fail(__FILE__, __LINE__,                              \
                             std::string(#a " != " #b " (") +                  \
                                 ::amp::to_str(amp_a_) + " vs " +            \
                                 ::amp::to_str(amp_b_) + ")");               \
        }                                                                     \
    } while (0)

#define AMP_CHECK_NEAR(a, b, tol)                                             \
    do {                                                                      \
        const double amp_a_ = (double) (a);                                   \
        const double amp_b_ = (double) (b);                                   \
        const double amp_d_ = amp_a_ > amp_b_ ? amp_a_ - amp_b_ : amp_b_ - amp_a_; \
        if (!(amp_d_ <= (double) (tol))) {                                    \
            ::amp_test::fail(__FILE__, __LINE__,                              \
                             std::string(#a " != " #b " (") +                 \
                                 ::amp::to_str(amp_a_) + " vs " +            \
                                 ::amp::to_str(amp_b_) + ")");               \
        }                                                                     \
    } while (0)
