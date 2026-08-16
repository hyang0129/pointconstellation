#!/usr/bin/env bash
set -euo pipefail

tools_dir="${POINTCONSTELLATION_MPEG_TOOLS_DIR:-artifacts/tools}"
tmc13_commit="a3d15c5e73bae20fbe2ec79be60994038a66dc8d"
dmetric_commit="bd6df59f7a6e1176706a88b3531c9be7f5db086f"
if [[ -n "${CMAKE:-}" ]]; then
  cmake_command=("$CMAKE")
elif command -v cmake >/dev/null 2>&1; then
  cmake_command=(cmake)
elif command -v uvx >/dev/null 2>&1; then
  cmake_command=(uvx --from cmake cmake)
else
  echo "cmake or uvx is required (for example: brew install cmake)" >&2
  exit 1
fi

mkdir -p "$tools_dir"

clone_pinned() {
  local url="$1"
  local path="$2"
  local commit="$3"
  if [[ ! -d "$path/.git" ]]; then
    git clone "$url" "$path"
    git -C "$path" checkout --detach "$commit"
  fi
  local actual
  actual="$(git -C "$path" rev-parse HEAD)"
  if [[ "$actual" != "$commit" ]]; then
    echo "$path is at $actual; expected pinned commit $commit" >&2
    exit 1
  fi
}

clone_pinned \
  https://github.com/MPEGGroup/mpeg-pcc-tmc13.git \
  "$tools_dir/mpeg-pcc-tmc13" \
  "$tmc13_commit"
clone_pinned \
  https://github.com/MPEGGroup/mpeg-pcc-dmetric.git \
  "$tools_dir/mpeg-pcc-dmetric" \
  "$dmetric_commit"

if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  patch_path="$(pwd)/scripts/patches/dmetric-apple-arm64.patch"
  if git -C "$tools_dir/mpeg-pcc-dmetric" \
    apply --check "$patch_path" 2>/dev/null; then
    git -C "$tools_dir/mpeg-pcc-dmetric" apply "$patch_path"
  elif ! git -C "$tools_dir/mpeg-pcc-dmetric" \
    apply --reverse --check "$patch_path" 2>/dev/null; then
    echo "the pc_error ARM portability patch cannot be applied cleanly" >&2
    exit 1
  fi
fi

"${cmake_command[@]}" \
  -S "$tools_dir/mpeg-pcc-tmc13" \
  -B "$tools_dir/mpeg-pcc-tmc13/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"${cmake_command[@]}" \
  --build "$tools_dir/mpeg-pcc-tmc13/build" \
  --config Release \
  --parallel

"${cmake_command[@]}" \
  -S "$tools_dir/mpeg-pcc-dmetric/source" \
  -B "$tools_dir/mpeg-pcc-dmetric/build/Release" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5
"${cmake_command[@]}" \
  --build "$tools_dir/mpeg-pcc-dmetric/build/Release" \
  --config Release \
  --parallel

echo "tmc3: $tools_dir/mpeg-pcc-tmc13/build/tmc3/tmc3"
echo "pc_error: $tools_dir/mpeg-pcc-dmetric/build/Release/pc_error"
