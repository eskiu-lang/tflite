#!/bin/sh
# Downloads the TensorFlow Lite 2.21.0 C library for this platform into DIR/lib
# (default ~/.eskiu/tflite-2.21.0) and prints DIR.
#
# Google publishes no desktop build of libtensorflowlite_c, so this takes the one in
# the Unity package of asus4/tf-lite-unity-sample (Apache-2.0), which builds
# TensorFlow's own tensorflow/lite/c:tensorflowlite_c target with Bazel for macOS
# (arm64 and x86_64), Linux (x86_64 and arm64) and Windows (x86_64), with the Metal
# (macOS) and OpenCL/OpenGL (Linux x86_64) GPU delegates. The archive is pinned by
# its SHA-256.
#
# Usage: tools/get_tflite.sh [DIR]
set -e
version=2.21.0
url=https://github.com/asus4/tf-lite-unity-sample/releases/download/v2.21.0-rc1/com.github.asus4.tflite-2.21.0-rc1.tgz
sha256=157d1936cd683aa8e1775f9b6f5ae2dafb652f4af9555f969b8563934133b645
dir=${1:-$HOME/.eskiu/tflite-$version}

case "$(uname -s)" in
Darwin)
    files="macOS/libtensorflowlite_c.dylib macOS/libtensorflowlite_metal_delegate.dylib"
    ;;
Linux)
    case "$(uname -m)" in
    x86_64 | amd64) files="Linux/x86_64/libtensorflowlite_c.so Linux/x86_64/libtensorflowlite_gpu_delegate.so" ;;
    aarch64 | arm64) files="Linux/arm64/libtensorflowlite_c.so" ;;
    *)
        echo "get_tflite: no prebuilt library for Linux $(uname -m)" >&2
        exit 1
        ;;
    esac
    ;;
MINGW* | MSYS* | CYGWIN*)
    files="Windows/libtensorflowlite_c.dll"
    ;;
*)
    echo "get_tflite: no prebuilt library for $(uname -s)" >&2
    exit 1
    ;;
esac

# The Windows DLL names itself tensorflowlite_c.dll, so programs look for that file.
target() {
    case "$1" in
    Windows/*) echo tensorflowlite_c.dll ;;
    *) basename "$1" ;;
    esac
}

have() {
    for f in $files; do
        [ -f "$dir/lib/$(target "$f")" ] || return 1
    done
}
if have && [ -f "$dir/VERSION" ] && [ "$(cat "$dir/VERSION")" = "$version" ]; then
    echo "$dir"
    exit 0
fi

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/tflite.tgz" "$url"
if command -v sha256sum > /dev/null 2>&1; then
    got=$(sha256sum "$tmp/tflite.tgz" | cut -d' ' -f1)
else
    got=$(shasum -a 256 "$tmp/tflite.tgz" | cut -d' ' -f1)
fi
if [ "$got" != "$sha256" ]; then
    echo "get_tflite: checksum mismatch for $url" >&2
    echo "  expected $sha256" >&2
    echo "  got      $got" >&2
    exit 1
fi
members=""
for f in $files; do
    members="$members package/Plugins/$f"
done
# shellcheck disable=SC2086
tar xzf "$tmp/tflite.tgz" -C "$tmp" $members package/LICENSE
mkdir -p "$dir/lib"
for f in $files; do
    cp "$tmp/package/Plugins/$f" "$dir/lib/$(target "$f")"
    chmod 755 "$dir/lib/$(target "$f")"
done
cp "$tmp/package/LICENSE" "$dir/LICENSE"
echo "$version" > "$dir/VERSION"
echo "$dir"
