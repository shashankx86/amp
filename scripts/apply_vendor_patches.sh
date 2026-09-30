#!/usr/bin/env bash
# Apply amp's patches to the vendored llama.cpp, idempotently.
#
# third_party/llama.cpp is gitignored and pinned by third_party/DEPS.lock, so an edit made
# directly in that tree is not version controlled and disappears on a re-fetch. Everything amp
# needs in llama.cpp therefore lives as a patch in third_party/patches and is applied by this
# script, which both scripts/fetch_deps.sh and the CMake configure step call so there is one
# implementation and a plain `cmake -S . -B build` cannot silently build an unpatched tree.
#
# Each patch is applied in filename order and must be independent of the others, because a tree
# can be found with any prefix of them already applied. A patch that neither applies nor is
# already applied is a real conflict and stops the build rather than producing a tree that is
# half patched.
set -euo pipefail
cd "$(dirname "$0")/.."

VENDOR=third_party/llama.cpp
if [ ! -d "$VENDOR/.git" ]; then
  echo "apply_vendor_patches: $VENDOR is not a git checkout; run scripts/fetch_deps.sh first" >&2
  exit 1
fi

shopt -s nullglob
# Absolute, because git -C resolves a patch path relative to the vendor directory, not
# to the directory this script was invoked from.
for p in "$PWD"/third_party/patches/*.patch; do
  pname=$(basename "$p" .patch)
  if git -C "$VENDOR" apply --reverse --check "$p" 2>/dev/null; then
    echo "apply_vendor_patches: $pname already applied"
  elif git -C "$VENDOR" apply --check "$p" 2>/dev/null; then
    git -C "$VENDOR" apply "$p"
    echo "apply_vendor_patches: applied $pname"
  else
    echo "apply_vendor_patches: $pname does not apply to the pinned tree at" \
         "$(git -C "$VENDOR" rev-parse --short HEAD); refusing to continue" >&2
    exit 1
  fi
done
