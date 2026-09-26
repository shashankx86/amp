// Read-only mmap of a model file. This is the only way amp ever touches weights:
// copying 14.65 GB into anonymous RAM froze the user's machine once, and mmap'd
// file-backed pages are evictable and re-readable, which is what makes the whole
// design work.
#pragma once

#include <cstdint>
#include <string>

#include "amp/status.h"

namespace amp {

class MappedFile {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile &)             = delete;
    MappedFile & operator=(const MappedFile &) = delete;
    MappedFile(MappedFile && other) noexcept;
    MappedFile & operator=(MappedFile && other) noexcept;

    static Result<MappedFile> open_read(const std::string & path);

    const uint8_t * data() const { return (const uint8_t *) data_; }
    size_t          size() const { return size_; }
    int             fd() const { return fd_; }
    const std::string & path() const { return path_; }

    // Blocking read into a caller buffer (used for synchronous warm-up and tests).
    Status pread_exact(uint64_t offset, void * dst, size_t len) const;

    // Fraction of pages in [offset, offset+len) currently resident in the page cache.
    // This is how amp avoids re-issuing readahead for data it already has: one mincore()
    // syscall per range is far cheaper than re-reading 1.3 MiB to find out.
    Result<double> resident_fraction(uint64_t offset, uint64_t length) const;

    // Hint the kernel that [offset, offset+len) will be read soon.
    Status advise_willneed(uint64_t offset, uint64_t len) const;
    Status advise_random(uint64_t offset, uint64_t len) const;
    Status drop_cache(uint64_t offset, uint64_t len) const;

private:
    const void * data_ = nullptr;
    size_t        size_ = 0;
    int           fd_   = -1;
    std::string   path_;
};

} // namespace amp
