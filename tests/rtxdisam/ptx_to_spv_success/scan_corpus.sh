#!/usr/bin/env bash
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="$HERE/../../tools/dump_isa_102/acuda_run"
ACUDA_ROOT="${ACUDA_ROOT:-/mnt/c/Users/Magnus/source/repos/ACuda}"
CORPUS="$ACUDA_ROOT/tests/zluda-ptx"

if [ ! -x "$BIN" ]; then
    echo "missing $BIN" >&2
    exit 2
fi
if [ ! -d "$CORPUS" ]; then
    echo "missing $CORPUS" >&2
    exit 2
fi

tmp="$(mktemp "${TMPDIR:-/tmp}/acuda_scan_XXXXXX.spv")"
errfile="$(mktemp "${TMPDIR:-/tmp}/acuda_scan_XXXXXX.err")"
trap 'rm -f "$tmp" "$errfile"' EXIT

ok=0
fail=0
skip=0
for ptx in "$CORPUS"/*.ptx; do
    name="$(basename "$ptx")"
    entry="$(sed -n 's/.*\.visible[[:space:]]\+\.entry[[:space:]]\+\([^ (]*\).*/\1/p' "$ptx" | head -n1)"
    if [ -z "$entry" ]; then
        echo "SKIP $name (no visible entry)"
        skip=$((skip + 1))
        continue
    fi
    if "$BIN" --dump "$tmp" "$ptx" "$entry" >"$errfile" 2>&1 &&
       { ! command -v spirv-val >/dev/null 2>&1 || spirv-val "$tmp" >"$errfile" 2>&1; }; then
        echo "OK $name"
        ok=$((ok + 1))
    else
        err="$(head -n1 "$errfile")"
        echo "FAIL $name: $err"
        fail=$((fail + 1))
    fi
done

echo "SUMMARY ok=$ok fail=$fail skip=$skip"
