#!/usr/bin/env bash
# Fetch amp's vendored dependencies, pinned by third_party/DEPS.lock.
#
# amp vendors llama.cpp rather than pointing at a checkout elsewhere: the point is to be
# standalone, with one clone and one build directory. This script is idempotent - re-running it
# only re-checks the pin.
set -euo pipefail
cd "$(dirname "$0")/.."

LOCK=third_party/DEPS.lock
[ -f "$LOCK" ] || { echo "fetch_deps: $LOCK missing"; exit 1; }

fetch_repo () {
  local name=$1 url=$2 sha=$3 dest=$4
  if [ -d "$dest/.git" ]; then
    local have
    have=$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo none)
    if [ "$have" = "$sha" ]; then
      echo "fetch_deps: $name already at ${sha:0:7}"
      return 0
    fi
    echo "fetch_deps: $name is at ${have:0:7}, pin wants ${sha:0:7} - updating"
    git -C "$dest" fetch --depth 1 origin "$sha"
    git -C "$dest" checkout --detach "$sha"
    return 0
  fi
  echo "fetch_deps: cloning $name ${sha:0:7}"
  rm -rf "$dest"
  mkdir -p "$dest"
  git -C "$dest" init -q
  git -C "$dest" remote add origin "$url"
  # Fetch just the pinned commit: shallow, and independent of branch naming.
  git -C "$dest" fetch -q --depth 1 origin "$sha"
  git -C "$dest" checkout -q --detach FETCH_HEAD
}

while read -r name url sha; do
  case "$name" in
    ''|\#*) continue ;;
  esac
  fetch_repo "$name" "$url" "$sha" "third_party/$name"
done < "$LOCK"

echo "fetch_deps: ok"
