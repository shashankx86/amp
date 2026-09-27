#!/usr/bin/env python3
"""Evict a file's clean page-cache pages, so cold-cache behaviour is reproducible.

    python3 scripts/evict_model_cache.py /path/to/model.gguf

Why this exists
---------------

`/proc/sys/vm/drop_caches` needs root, and this box has no sudo. But a model's
weights are mapped read-only and are therefore *clean* pages, and the kernel is
happy to drop clean pages for a specific file when asked. `posix_fadvise(
POSIX_FADV_DONTNEED)` does exactly that, unprivileged.

That matters because steady-state decode is cache-warm, so a benchmark run that
never leaves the warm state cannot tell you anything about cold behaviour. The
project's own reason for existing is "does not collapse when the page cache is
under pressure", and that is a cold-cache property. Measuring it needs a
reproducible cold state, which is what this provides.

What it does *not* do: evict dirty pages. If the model file has unwritten
contents they stay, and the reported eviction can be smaller than the file size.
Always check the reported numbers rather than assuming the whole file went.
"""

import ctypes
import ctypes.util
import os
import sys

# From linux/fcntl.h. Not exposed by the os module.
POSIX_FADV_DONTNEED = 4


def main():
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} FILE", file=sys.stderr)
        return 2

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"no such file: {path}", file=sys.stderr)
        return 2

    size = os.path.getsize(path)
    libc_name = ctypes.util.find_library("c")
    if not libc_name:
        print("cannot locate libc", file=sys.stderr)
        return 1
    libc = ctypes.CDLL(libc_name, use_errno=True)

    fd = os.open(path, os.O_RDONLY)
    try:
        # 0, 0 means "the whole file".
        rc = libc.posix_fadvise(ctypes.c_int(fd), ctypes.c_long(0),
                                ctypes.c_long(0), ctypes.c_int(POSIX_FADV_DONTNEED))
    finally:
        os.close(fd)

    if rc != 0:
        print(f"posix_fadvise failed: errno {rc}", file=sys.stderr)
        return 1

    print(f"advised DONTNEED: {path} ({size / 2**30:.2f} GiB)")
    print("dirty pages, if any, remain resident; verify with a profile run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
