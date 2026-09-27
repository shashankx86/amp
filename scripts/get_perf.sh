#!/usr/bin/env bash
# Make `perf` available without root, so the decode profile stays reproducible.
#
#     ./scripts/get_perf.sh            # extract into build/perf and print the path to use
#     eval "$(./scripts/get_perf.sh --print-path)"
#
# Why this exists rather than "just install perf":
#
#   * There is no sudo on this box, so `pacman -S perf` needs a password.
#   * The package is usually *already* in /var/cache/pacman/pkg, because something pulled it in
#     before. Extracting a .pkg.tar.zst with bsdtar needs no privileges at all and touches nothing
#     system-wide, so nothing is installed and nothing needs undoing.
#   * Everything lands under build/, which is gitignored, so this cannot dirty the tree.
#
# What is verified before this is worth trusting:
#
#   * The kernel has the perf subsystem: CONFIG_PERF_EVENTS=y. Check with
#     `zcat /proc/config.gz | grep CONFIG_PERF_EVENTS` or /boot/config-$(uname -r).
#   * perf_event_open() actually works for this user. paranoid=2 still permits sampling your own
#     process in user space; it only forbids kernel-space and system-wide capture. Verify before
#     concluding that a profile is impossible, because "perf is not installed" and "perf cannot
#     work here" are different problems with different fixes.
#   * A profile is a sampling profile. Without root you cannot do `perf record -a`, capture kernel
#     stacks, or use raw tracepoints. For per-op attribution inside ggml, symbol-level sampling of
#     our own process is enough, provided the binary is not stripped. Check with
#     `nm -C build/bin/amp-server | grep -c 'ggml_compute_forward_'`; a Release build keeps the
#     symbol table, so no -g rebuild is needed.
set -euo pipefail
cd "$(dirname "$0")/.."

DEST="build/perf"
PRINT_PATH=0
[ "${1:-}" = "--print-path" ] && PRINT_PATH=1

if [ "$PRINT_PATH" = "1" ]; then
    echo "export PATH=\"$PWD/$DEST/root/usr/bin:\$PATH\""
    echo "export LD_LIBRARY_PATH=\"$PWD/$DEST/root/usr/lib:\${LD_LIBRARY_PATH:-}\""
    exit 0
fi

# ---- gate 1: can the kernel do perf events at all
#
# Note two traps, both hit while writing this. /boot/config-$(uname -r) is often present but
# unreadable (mode 0600 root), so `[ -r ]` is the right test and falling through is not an error.
# And `zcat ... | grep -q` fails under `set -o pipefail`, because grep -q exits on the first match
# and SIGPIPEs zcat, which then reports failure. Capture first, then grep the variable.
KCFG=""
if [ -r "/boot/config-$(uname -r)" ]; then
    KCFG="/boot/config-$(uname -r)"
elif [ -r /proc/config.gz ]; then
    KCFG=/proc/config.gz
fi

if [ -n "$KCFG" ]; then
    case "$KCFG" in
        *.gz) CFG="$(zcat "$KCFG")" ;;
        *)     CFG="$(cat "$KCFG")" ;;
    esac
    # A herestring, not a pipe. Under `set -o pipefail` any `producer | grep -q` fails, because
    # grep -q exits on the first match, the producer takes SIGPIPE, and the pipeline reports the
    # producer's failure. That is a false negative on a kernel that plainly supports perf.
    if ! grep -q '^CONFIG_PERF_EVENTS=y' <<< "$CFG"; then
        echo "get_perf: $KCFG has CONFIG_PERF_EVENTS unset; perf cannot work here" >&2
        exit 1
    fi
    echo "get_perf: $KCFG confirms CONFIG_PERF_EVENTS=y"
else
    echo "get_perf: warning: no readable kernel config, cannot verify CONFIG_PERF_EVENTS" >&2
fi

# ---- gate 2: is the package already in the pacman cache
PKG="$(ls -1 /var/cache/pacman/pkg/perf-*.pkg.tar.zst 2>/dev/null | tail -1 || true)"
if [ -z "$PKG" ]; then
    if command -v pacman >/dev/null && [ -d /var/lib/pacman/sync ]; then
        # -Sp prints the URL without downloading or installing, and needs no privileges.
        URL="$(pacman -Sp perf 2>/dev/null | grep -E '^https?://' | tail -1 || true)"
    fi
    if [ -n "${URL:-}" ]; then
        echo "get_perf: fetching $URL"
        mkdir -p build/perf/dl
        curl -fsSL -o build/perf/dl/perf.pkg.tar.zst "$URL"
        PKG="build/perf/dl/perf.pkg.tar.zst"
    else
        cat >&2 <<'MSG'
get_perf: no perf package in /var/cache/pacman/pkg and no download URL available.

  Install it with a password:   sudo pacman -S perf
  or fetch it by hand from https://archlinux.org/packages/extra/x86_64/perf/
MSG
        exit 1
    fi
fi

echo "get_perf: using $PKG"
rm -rf "$DEST"
mkdir -p "$DEST/root"
bsdtar -xf "$PKG" -C "$DEST/root"

PERF="$PWD/$DEST/root/usr/bin/perf"
[ -x "$PERF" ] || { echo "get_perf: extraction did not produce $PERF" >&2; exit 1; }
chmod +x "$PERF" 2>/dev/null || true

export LD_LIBRARY_PATH="$PWD/$DEST/root/usr/lib:${LD_LIBRARY_PATH:-}"
"$PERF" --version || { echo "get_perf: extracted perf does not run" >&2; exit 1; }

# ---- gate 3: confirm the binary can be attributed, or the profile will be useless
SYMS="$(nm -C build/bin/amp-server 2>/dev/null | grep -c 'ggml_compute_forward_' || true)"
if [ "${SYMS:-0}" -lt 10 ]; then
    echo "get_perf: warning: only $SYMS ggml_compute_forward_* symbols in amp-server." >&2
    echo "          A profile will not attribute per op. Rebuild with -g (RelWithDebInfo)." >&2
else
    echo "get_perf: $SYMS ggml_compute_forward_* symbols present, per-op attribution will work"
fi

cat <<MSG

  export PATH="$PWD/$DEST/root/usr/bin:\$PATH"
  export LD_LIBRARY_PATH="$PWD/$DEST/root/usr/lib:\${LD_LIBRARY_PATH:-}"

Then, with a decode request in flight:
  perf record -F 999 -p \$(pgrep -x amp-server) -o build/decode.data -- sleep 60
  perf report -i build/decode.data --stdio --no-children
MSG
