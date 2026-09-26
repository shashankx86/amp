// Tests for the reasoning round-trip. The failure this prevents is silent and expensive: a
// double-tagged reasoning block makes the re-rendered prompt differ from the cached one, and every
// turn then re-evaluates the whole conversation.
#include "test_harness.h"

#include "amp/util/text.h"

using namespace amp;

AMP_TEST(normalize_reasoning_strips_wrapping_tags) {
    // The well-behaved client: the server reported the inner text, so there is nothing to strip.
    AMP_CHECK_EQ(normalize_reasoning("step one, then two"), std::string("step one, then two"));
    // A client that echoes the tagged form must not get it wrapped a second time by the template.
    AMP_CHECK_EQ(normalize_reasoning("<think>hmm</think>"), std::string("hmm"));
    AMP_CHECK_EQ(normalize_reasoning(" <think>\nhmm\n</think> "), std::string("hmm"));
    // Only the wrapping pair goes; prose that mentions a tag survives.
    AMP_CHECK_EQ(normalize_reasoning("<think>a</think> and <think>b</think>"),
                 std::string("a</think> and <think>b"));
    // Multiline reasoning, which is the normal case.
    const std::string big = "<think>\n1. first\n2. second\n</think>";
    AMP_CHECK_EQ(normalize_reasoning(big), std::string("1. first\n2. second"));
}

AMP_TEST(normalize_reasoning_is_identity_for_empty_and_tagless) {
    AMP_CHECK_EQ(normalize_reasoning(""), std::string(""));
    AMP_CHECK_EQ(normalize_reasoning("   \n "), std::string(""));
    AMP_CHECK_EQ(normalize_reasoning("no tags here"), std::string("no tags here"));
    // A bare start tag with nothing after it: stripping leaves nothing, which is correct.
    AMP_CHECK_EQ(normalize_reasoning("<think>"), std::string(""));
}

AMP_TEST(normalize_reasoning_round_trips_through_the_template_shape) {
    // What the server stores -> what the client echoes -> what the server stores again. This has to
    // be a fixed point, or the cache diverges on every turn.
    const std::string stored = "consider the options";
    const std::string client_echo = stored;   // the common, correct case
    AMP_CHECK_EQ(normalize_reasoning(client_echo), stored);
    // And a client that re-wraps still lands on the same value.
    const std::string wrapped = std::string("<think>\n") + stored + "\n</think>";
    AMP_CHECK_EQ(normalize_reasoning(wrapped), stored);
}
