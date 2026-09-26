// OpenAI-compatible HTTP surface for amp.
//
// Routes:
//   GET  /health              liveness + a one-line status
//   GET  /v1/models           model list
//   GET  /props               amp diagnostics: the plan, the cache, VRAM/cache state
//   POST /tokenize            token ids for a prompt (diagnostics and parity work)
//   POST /v1/completions       text completion   (alias: /completion)
//   POST /v1/chat/completions chat completion    (alias: /chat/completions)
#pragma once

#include <functional>
#include <string>

#include "amp/server/http_server.h"
#include "amp/server/service.h"

namespace amp {

class OpenAIApi {
public:
    // Registers all routes on `srv`. The service must outlive the server.
    static void register_routes(http::Server & srv, InferenceService & svc,
                                const ServerConfig & cfg);

private:
    static Status health(const http::Request & req, http::ResponseWriter & w,
                               InferenceService & svc);
    static Status models(const http::Request & req, http::ResponseWriter & w,
                               InferenceService & svc, const ServerConfig & cfg);
    static Status props(const http::Request & req, http::ResponseWriter & w,
                              InferenceService & svc);
    static Status tokenize(const http::Request & req, http::ResponseWriter & w,
                                 InferenceService & svc);
    static Status completions(const http::Request & req, http::ResponseWriter & w,
                                    InferenceService & svc, const ServerConfig & cfg);
    static Status chat_completions(const http::Request & req, http::ResponseWriter & w,
                                         InferenceService & svc, const ServerConfig & cfg);
};

} // namespace amp
