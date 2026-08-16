#!/usr/bin/env bash
set -euo pipefail

source_url="https://modelnet.cs.princeton.edu/ModelNet40.zip"
expected_sha256="42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac"
archive_path="data/downloads/ModelNet40.zip"
extract_parent="data/modelnet40_official"
dataset_root="$extract_parent/ModelNet40"

if [[ "${1:-}" != "--accept-academic-use" ]]; then
  echo "ModelNet40 is provided for academic research only." >&2
  echo "Review https://modelnet.cs.princeton.edu/download.html" >&2
  echo "Then rerun with --accept-academic-use if those terms apply." >&2
  exit 2
fi

if ! command -v curl >/dev/null 2>&1 \
  || ! command -v unzip >/dev/null 2>&1 \
  || ! command -v zipinfo >/dev/null 2>&1; then
  echo "curl, unzip, and zipinfo are required" >&2
  exit 1
fi

hash_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  elif command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    echo "shasum or sha256sum is required" >&2
    exit 1
  fi
}

mkdir -p "$(dirname "$archive_path")" "$extract_parent"
if [[ ! -f "$archive_path" ]] \
  || [[ "$(hash_file "$archive_path")" != "$expected_sha256" ]]; then
  curl \
    --location \
    --fail \
    --retry 3 \
    --continue-at - \
    --output "$archive_path" \
    "$source_url"
fi

actual_sha256="$(hash_file "$archive_path")"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "ModelNet40 archive hash mismatch: $actual_sha256" >&2
  exit 1
fi

if zipinfo -1 "$archive_path" | awk '
  /^\// || /(^|\/)\.\.(\/|$)/ { unsafe = 1 }
  END { exit unsafe ? 0 : 1 }
'; then
  echo "ModelNet40 archive contains an unsafe path" >&2
  exit 1
fi

unzip -q -n "$archive_path" -d "$extract_parent"
train_count="$(find "$dataset_root" -path '*/train/*.off' -type f | wc -l | tr -d ' ')"
test_count="$(find "$dataset_root" -path '*/test/*.off' -type f | wc -l | tr -d ' ')"
if [[ "$train_count" != "9843" || "$test_count" != "2468" ]]; then
  echo "unexpected ModelNet40 counts: train=$train_count test=$test_count" >&2
  exit 1
fi

echo "ModelNet40 ready at $dataset_root"
echo "archive sha256: $actual_sha256"
echo "official split counts: train=$train_count test=$test_count"
