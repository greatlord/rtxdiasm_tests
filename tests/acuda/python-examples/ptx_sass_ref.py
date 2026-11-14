"""PTX validity cross-check with NVIDIA's own toolchain (reference loop).

For every tests/ptx/*.ptx not marked ACUDA-EXTENSION: assemble with real
ptxas (sm_80) and dump SASS text with cuobjdump. This proves our supported subset is genuine
NVIDIA-accepted PTX (stronger than our own parser accepting it) and
refreshes the ground-truth corpus in tests/sass/.

It found real divergences already: mad.lo with a special-reg source
and %r<4> meaning %r0-%r3 (both fixed in our parser/test kernel).

Tools come from pip venvs (not system packages); override via env:
    ACUDA_PTXAS=/path/to/ptxas ACUDA_CUOBJDUMP=/path/to/cuobjdump
Missing tools -> skip with exit 0 (never fail the build).

Usage (from ACuda/):
    "/home/magnusolsen/git/aicode-ptx/virtual python/ven-cu134/bin/python" python-test/ptx_sass_ref.py [--refresh]
--refresh rewrites tests/sass/*.sm_80.sass.txt from current output.
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PTX_DIR = os.path.join(ROOT, "tests", "ptx")
SASS_DIR = os.path.join(ROOT, "tests", "sass")


def find_tool(env, names):
    override = os.environ.get(env)
    if override:
        return override if os.path.exists(override) else None
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def main():
    refresh = "--refresh" in sys.argv
    ptxas = find_tool("ACUDA_PTXAS", ["ptxas"])
    cuobjdump = find_tool("ACUDA_CUOBJDUMP", ["cuobjdump"])
    if not ptxas or not cuobjdump:
        print("ptx_sass_ref: SKIP (ptxas/cuobjdump not found; "
              "set ACUDA_PTXAS/ACUDA_CUOBJDUMP)")
        return 0
    print(f"ptxas: {ptxas}")
    print(f"cuobjdump: {cuobjdump}")
    fails = 0
    for name in sorted(os.listdir(PTX_DIR)):
        if not name.endswith(".ptx"):
            continue
        base = name[:-4]
        src = os.path.join(PTX_DIR, name)
        with open(src, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(4096)
        if "ACUDA-EXTENSION" in head:
            print(f"skip {name}: ACuda extension (not valid PTX for ptxas)")
            continue
        tm = re.search(r"^\s*\.target\s+sm_([0-9]+)", head,
                       re.MULTILINE)
        # Use the file's target when ptxas supports it (sm_80+); old
        # sm_30/sm_35 sources stay assembled at sm_80 (forward-compatible),
        # avoiding ptxas's dropped-arch errors.
        arch = "sm_80"
        if tm and int(tm.group(1)) >= 80:
            arch = f"sm_{tm.group(1)}"
        outdir = os.path.join(ROOT, "build", "sassref")
        os.makedirs(outdir, exist_ok=True)
        cubin = os.path.join(outdir, base + ".sm_80.cubin")
        r = subprocess.run([ptxas, f"-arch={arch}", "-m64", src,
                            "-o", cubin], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL ptxas {name}:\n{r.stderr or r.stdout}")
            fails += 1
            continue
        r = subprocess.run([cuobjdump, "-sass", cubin],
                           capture_output=True, text=True)
        sass = r.stdout
        nfunc = sass.count("Function :")
        print(f"ok {name}: {nfunc} function(s)")
        if refresh and nfunc > 0:
            with open(os.path.join(SASS_DIR, base + ".sm_80.sass.txt"),
                      "w") as f:
                f.write(sass)
    if fails:
        raise SystemExit(f"FAIL: {fails} kernel(s) rejected by ptxas")
    print("ptx_sass_ref passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
