#!/usr/bin/env bash
set -euo pipefail

draco_release="1.5.7"
draco_commit="8786740086a9f4d83f44aa83badfbea4dce7a1b5"
draco_url="https://github.com/google/draco.git"
external_root="artifacts/external/draco"

usage() {
  cat <<'EOF'
Usage: scripts/build_draco.sh [--root PATH]

Builds the pinned Draco 1.5.7 command-line encoder and decoder, runs an actual
point-cloud round trip, and records binary hashes. On Apple ARM, the build is
forced to arm64 and the resulting Mach-O architecture is verified.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      external_root="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$external_root"
source_dir="$external_root/source"
build_dir="$external_root/build"
if [[ ! -d "$source_dir/.git" ]]; then
  if [[ -e "$source_dir" ]]; then
    echo "error: $source_dir exists but is not a Git checkout" >&2
    exit 1
  fi
  git clone --filter=blob:none "$draco_url" "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$draco_commit"
git -C "$source_dir" checkout --detach "$draco_commit"
actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$draco_commit" ]]; then
  echo "error: Draco checkout verification failed" >&2
  exit 1
fi

cmake_options=(
  -DCMAKE_BUILD_TYPE=Release
  -DDRACO_BUILD_EXECUTABLES=ON
  -DDRACO_TESTS=OFF
  -DDRACO_POINT_CLOUD_COMPRESSION=ON
)
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  cmake_options+=(-DCMAKE_OSX_ARCHITECTURES=arm64)
fi
cmake -S "$source_dir" -B "$build_dir" "${cmake_options[@]}"
cmake --build "$build_dir" --config Release --parallel

encoder=$(find "$build_dir" -type f -name draco_encoder -perm -111 -print -quit)
decoder=$(find "$build_dir" -type f -name draco_decoder -perm -111 -print -quit)
if [[ -z "$encoder" || -z "$decoder" ]]; then
  echo "error: Draco command-line tools were not built" >&2
  exit 1
fi
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  if ! file "$encoder" | grep -q 'arm64'; then
    echo "error: draco_encoder is not an Apple ARM binary" >&2
    exit 1
  fi
  if ! file "$decoder" | grep -q 'arm64'; then
    echo "error: draco_decoder is not an Apple ARM binary" >&2
    exit 1
  fi
fi

smoke_dir=$(mktemp -d "${TMPDIR:-/tmp}/pointconstellation-draco.XXXXXX")
cleanup() {
  if [[ "$smoke_dir" == "${TMPDIR:-/tmp}/pointconstellation-draco."* ]]; then
    rm -rf -- "$smoke_dir"
  fi
}
trap cleanup EXIT
cat > "$smoke_dir/input.ply" <<'PLY'
ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
end_header
-1 -1 -1
-0.25 0.5 0.75
0.5 -0.25 0
1 1 1
PLY
"$encoder" -point_cloud -i "$smoke_dir/input.ply" \
  -o "$smoke_dir/stream.drc" -qp 10 -cl 10
"$decoder" -i "$smoke_dir/stream.drc" -o "$smoke_dir/reconstruction.ply"
test -s "$smoke_dir/stream.drc"
test -s "$smoke_dir/reconstruction.ply"

python3 - "$external_root/build_manifest.json" "$encoder" "$decoder" \
  "$draco_release" "$draco_commit" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

output, encoder, decoder, release, commit = sys.argv[1:]

def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

manifest = {
    "release": release,
    "release_commit": commit,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "encoder_executable": str(Path(encoder).resolve()),
    "encoder_sha256": digest(encoder),
    "decoder_executable": str(Path(decoder).resolve()),
    "decoder_sha256": digest(decoder),
    "point_cloud_round_trip_verified": True,
}
Path(output).write_text(json.dumps(manifest, indent=2) + "\n")
PY

echo "Built and verified Draco $draco_release ($actual_commit)."
echo "Build manifest: $external_root/build_manifest.json"
