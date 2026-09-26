// Tests for the JSON layer: it is the API's wire format, so parse/dump symmetry and hostile input
// handling matter more than anything clever.
#include "test_harness.h"

#include "amp/format.h"
#include "amp/util/json.h"

using namespace amp;
using amp::json::Value;

AMP_TEST(json_parses_scalars) {
    AMP_CHECK(Value::parse("null").value_or(Value()).is_null());
    AMP_CHECK(Value::parse("true").value_or(Value()).as_bool() == true);
    AMP_CHECK(Value::parse("false").value_or(Value()).as_bool() == false);
    AMP_CHECK_EQ(Value::parse("42").value_or(Value()).as_int(), 42);
    AMP_CHECK_EQ(Value::parse("-17").value_or(Value()).as_int(), -17);
    AMP_CHECK_NEAR(Value::parse("3.5").value_or(Value()).as_double(), 3.5, 1e-12);
    AMP_CHECK_EQ(Value::parse("\"hi\"").value_or(Value()).as_string(), std::string("hi"));
}

AMP_TEST(json_parses_containers) {
    auto v = Value::parse("{\"a\":[1,2,3],\"b\":{\"c\":true},\"d\":\"x\"}");
    AMP_CHECK(v.ok());
    const Value & root = *v;
    AMP_CHECK_EQ(root.size(), 3u);
    AMP_CHECK(root.has("a") && root.get("a")->is_array());
    AMP_CHECK_EQ(root.get("a")->size(), 3u);
    AMP_CHECK_EQ(root.get("a")->at(1).as_int(), 2);
    AMP_CHECK(root.get("b") != nullptr && root.get("b")->get("c")->as_bool());
    AMP_CHECK_EQ(root.get("d")->as_string(), std::string("x"));
    AMP_CHECK(root.get("missing") == nullptr);
}

AMP_TEST(json_parses_nested_messages_shape) {
    // the exact shape OpenCode posts to /v1/chat/completions
    const std::string body =
        R"({"model":"amp","messages":[{"role":"user","content":"hi"},)"
        R"({"role":"assistant","content":null,"reasoning_text":"<think>x</think>"},)"
        R"({"role":"user","content":[{"type":"text","text":"part"}]}],"stream":true})";
    auto v = Value::parse(body);
    AMP_CHECK_MSG(v.ok(), v.message());
    const Value & msgs = *v->get("messages");
    AMP_CHECK_EQ(msgs.size(), 3u);
    AMP_CHECK_EQ(msgs.at(0).get("role")->as_string(), std::string("user"));
    // the alias llama.cpp ignores: this is the whole bug
    AMP_CHECK(msgs.at(1).get("reasoning_text") != nullptr);
    AMP_CHECK(msgs.at(1).get("reasoning_content") == nullptr);
    // content arrays. Note the explicit dereference: with implicit Value(bool) this used to bind a
    // temporary instead, which is exactly the bug the explicit constructors now prevent.
    const Value * content = msgs.at(2).get("content");
    AMP_CHECK(content != nullptr);
    AMP_CHECK(content->is_array());
    AMP_CHECK_EQ(content->at(0).get("text")->as_string(), std::string("part"));
    AMP_CHECK(v->get("stream")->as_bool());
}

AMP_TEST(json_string_escapes_round_trip) {
    const std::string awkward = "line\nbreak\ttab \"quoted\" back\\slash \x01 ctrl";
    Value v = Value::object();
    v.set("s", awkward);
    auto back = Value::parse(v.dump());
    AMP_CHECK(back.ok());
    AMP_CHECK_EQ(back->get("s")->as_string(), awkward);
}

AMP_TEST(json_unicode_escape) {
    auto v = Value::parse("\"\\u00e9\\u4e2d\"");
    AMP_CHECK(v.ok());
    // é (2 bytes) and 中 (3 bytes)
    AMP_CHECK_EQ(v->as_string().size(), 5u);
    auto back = Value::parse(v->dump());
    AMP_CHECK(back.ok());
    AMP_CHECK_EQ(back->as_string(), v->as_string());
}

AMP_TEST(json_rejects_garbage) {
    AMP_CHECK(!Value::parse("").ok());
    AMP_CHECK(!Value::parse("{").ok());
    AMP_CHECK(!Value::parse("{\"a\":}").ok());
    AMP_CHECK(!Value::parse("[1,2").ok());
    AMP_CHECK(!Value::parse("tru").ok());
    AMP_CHECK(!Value::parse("\"unterminated").ok());
    AMP_CHECK(!Value::parse("{} extra").ok());
    AMP_CHECK(!Value::parse("").ok());
}

AMP_TEST(json_rejects_deep_nesting) {
    std::string deep;
    for (int i = 0; i < 500; i++) {
        deep += "[";
    }
    AMP_CHECK_MSG(!Value::parse(deep).ok(), "unbounded nesting should be rejected");
}

AMP_TEST(json_dump_is_compact_and_pretty) {
    Value o = Value::object();
    o.set("n", (int64_t) 3);
    o.set("s", "x");
    Value a = Value::array();
    a.push((int64_t) 1);
    a.push("two");
    o.set("a", std::move(a));
    AMP_CHECK_EQ(o.dump(), std::string(R"({"a":[1,"two"],"n":3,"s":"x"})"));
    // pretty output must parse back to the same thing
    auto back = Value::parse(o.dump(2));
    AMP_CHECK(back.ok());
    AMP_CHECK_EQ(back->dump(), o.dump());
}

AMP_TEST(json_numbers_keep_precision) {
    Value v = Value::object();
    v.set("big", (int64_t) 9007199254740993ll);
    auto back = Value::parse(v.dump());
    AMP_CHECK(back.ok());
    AMP_CHECK_EQ(back->get("big")->as_int(), 9007199254740993ll);
    // doubles round-trip exactly
    Value d = Value::object();
    d.set("x", 0.1 + 0.2);
    auto db = Value::parse(d.dump());
    AMP_CHECK(db.ok());
    AMP_CHECK_NEAR(db->get("x")->as_double(), 0.1 + 0.2, 1e-15);
}

AMP_TEST(json_mutators) {
    Value v = Value::object();
    v["k"] = Value("v");
    AMP_CHECK_EQ(v.get("k")->as_string(), std::string("v"));
    Value arr = Value::array();
    arr.push(1);
    arr.push("two");
    AMP_CHECK_EQ(arr.as_array().size(), 2u);
    AMP_CHECK(arr.is_array());
    // type promotion on access
    Value t;
    t.as_array().push_back(Value(1));
    AMP_CHECK(t.is_array());
    t.as_object();
    AMP_CHECK(t.is_object());
}
