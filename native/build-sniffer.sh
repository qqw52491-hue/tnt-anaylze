#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NDK="${ANDROID_NDK_HOME:-$HOME/Library/Android/sdk/ndk/27.0.12077973}"
CC="$NDK/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang"
if [ ! -x "$CC" ]; then
    echo "找不到 Android NDK 编译器；请设置 ANDROID_NDK_HOME" >&2
    exit 1
fi
mkdir -p "$ROOT/target/android"
"$CC" -std=gnu11 -O2 -g -fPIC -shared -Wall -Wextra \
    -Wl,-z,defs -Wl,-soname,libtntsniff.so \
    "$ROOT/native/tnt_sniff.c" -o "$ROOT/target/android/libtntsniff.so.new" -llog -ldl -lm
mv "$ROOT/target/android/libtntsniff.so.new" "$ROOT/target/android/libtntsniff.so"
echo "已编译 $ROOT/target/android/libtntsniff.so (Qt SSL 明文采集 tls=3)"
