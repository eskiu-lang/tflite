#!/bin/sh
# Compares src/raw.esk with the C compiler's view of the TensorFlow Lite headers:
# every struct's size and field offsets and every enum value. The Eskiu program also
# takes the address of every extern, so linking it checks that the library exports
# each one.
# Usage: tools/layout_check.sh LIB_DIR [INCLUDE_DIR]
set -e
root=$(cd "$(dirname "$0")/.." && pwd)
lib=$1
inc=${2:-$root/tools/include}
cc=${CC:-cc}
eskiuc=${ESKIUC:-eskiuc}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
sed 's/^import <tflite\/raw>;//' "$root/tools/layout/layout.esk" > "$tmp/layout.esk"
case "$(uname -s)" in
Darwin)
    # metal_delegate.h imports Metal, an Objective-C header.
    "$cc" -x objective-c -I"$inc" "$root/tools/layout/layout.c" -o "$tmp/c"
    "$eskiuc" "$root/src/raw.esk" "$tmp/layout.esk" -o "$tmp/e" -L"$lib" --link-arg=-Wl,-rpath,"$lib"
    ;;
MINGW* | MSYS* | CYGWIN*)
    "$cc" -I"$inc" "$root/tools/layout/layout.c" -o "$tmp/c"
    # The program needs the DLL beside it.
    cp "$lib"/*.dll "$tmp/"
    "$eskiuc" "$root/src/raw.esk" "$tmp/layout.esk" -o "$tmp/e" -L"$lib"
    ;;
*)
    "$cc" -I"$inc" "$root/tools/layout/layout.c" -o "$tmp/c"
    "$eskiuc" "$root/src/raw.esk" "$tmp/layout.esk" -o "$tmp/e" -L"$lib" --link-arg=-Wl,-rpath,"$lib"
    ;;
esac
"$tmp/c" > "$tmp/c.txt"
"$tmp/e" > "$tmp/e.txt"
diff "$tmp/c.txt" "$tmp/e.txt"
echo "layout ok: $(wc -l < "$tmp/c.txt" | tr -d ' ') sizes, offsets, enum values and the function count"
