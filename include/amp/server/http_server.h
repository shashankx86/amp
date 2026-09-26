// Minimal HTTP/1.1 server for amp.
//
// Deliberately dependency-free: a listening socket, a small thread pool, and enough HTTP to serve
// an OpenAI-compatible API with SSE streaming. Written from scratch because adding a third-party
// HTTP stack to a project that vendors one C++ dependency would be the wrong trade for ~400 lines.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "amp/status.h"

namespace amp {
namespace http {

// A parsed request. Headers are lowercased; the body is kept raw for the JSON layer.
struct Request {
    std::string                          method;
    std::string                          path;
    std::string                          query;
    std::map<std::string, std::string>  headers;   // lowercased keys
    std::string                          body;
    int                                  content_length = 0;
    bool                                 keep_alive  = true;

    std::string header(const std::string & name, const std::string & def = "") const;
    bool        wants_stream() const;
    bool        is_chunked() const;
};

// Chunked writer used for both SSE and plain responses. The handler writes incrementally and can
// flush partial output, which streaming needs.
class ResponseWriter {
public:
    explicit ResponseWriter(int fd) : fd_(fd) {}

    Status send_headers(int status, const std::string & content_type, bool chunked,
                        const std::vector<std::pair<std::string, std::string>> & extra = {});
    Status write(const std::string & data);       // body chunk (chunked framing applied)
    Status write_sse(const std::string & data);   // "data: ...\n\n"
    Status write_sse_done();                      // "data: [DONE]\n\n"
    Status finish();                              // terminating chunk (or Content-Length body end)

    bool     headers_sent() const { return headers_sent_; }
    uint64_t bytes_written() const { return bytes_; }

private:
    Status write_raw(const char * data, size_t len);

    int         fd_            = -1;
    bool        headers_sent_  = false;
    bool        chunked_       = false;
    uint64_t    bytes_         = 0;
    std::string outbuf_;
};

// Handler: given a request and a writer, produce a response. Returning a failed Status after
// headers have been sent closes the connection rather than sending an error page.
using Handler = std::function<Status(const Request &, ResponseWriter &)>;

class Server {
public:
    ~Server();

    static Result<std::unique_ptr<Server>> listen(const std::string & host, uint16_t port,
                                                  int n_threads = 4);

    void route(const std::string & method, const std::string & path, Handler h);
    // Serve until stop() is called. Blocks.
    Status serve();
    void stop();

    uint16_t port() const { return port_; }
    const std::string & last_error() const { return last_error_; }

private:
    Status handle_connection(int fd);
    Status dispatch(const Request & req, ResponseWriter & w);
    static Status send_error(ResponseWriter & w, int status, const std::string & message);

    int                                      listen_fd_ = -1;
    uint16_t                                 port_      = 0;
    std::atomic<bool>                        stop_{false};
    std::map<std::string, Handler>            routes_;   // key = "METHOD path"
    std::vector<std::string>                  order_;    // registration order, for logging
    std::string                              last_error_;
};

// ---- tiny helpers shared with the JSON layer ----
std::string url_decode(const std::string & s);
std::string json_escape(const std::string & s);
// Minimal event-stream framing helper.
std::string sse_event(const std::string & data, const std::string & event = "");

} // namespace http
} // namespace amp
