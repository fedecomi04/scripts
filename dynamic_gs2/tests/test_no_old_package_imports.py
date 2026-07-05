"""test_no_old_package_imports.py — permanent detach regression gate.

Fails if ANY dynamic_gs2 source file has a load-bearing reference to the old
`dynamic_gs` package: a `from dynamic_gs.` / `import dynamic_gs.` statement, or a
subprocess `-m dynamic_gs.` invocation. Provenance comments/docstrings that merely
MENTION the old package (e.g. "inlined verbatim from dynamic_gs.utils.X") are allowed —
the gate only trips on real code that would break once `dynamic_gs/` is removed.

The two verify/ab_*_phase0b.py files are allowlisted: they import the OLD package on
purpose as the A/B correctness reference (they compare gs2's port against the frozen
original). They are not on any runtime path.

Run (from scripts/):  conda run -n dynamic_gs python -m dynamic_gs2.tests.test_no_old_package_imports
"""
import re
import sys
from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parents[1]              # scripts/dynamic_gs2/
_ALLOWLIST = {
    "verify/ab_isolated_phase0b.py",
    "verify/ab_unit_phase0b.py",
}

# A load-bearing reference: an import statement or a subprocess -m string targeting
# the old package. We strip line comments and skip triple-quoted docstring bodies so
# provenance notes ("inlined from dynamic_gs.utils.X") don't false-trip.
_IMPORT_RE = re.compile(r"^\s*(from|import)\s+dynamic_gs\.")
_SUBPROC_RE = re.compile(r"""["']-m["']\s*,\s*["']dynamic_gs\.""")


def _code_lines(text: str):
    """Yield (lineno, line) for real code lines, skipping triple-quoted string bodies
    and stripping trailing line comments."""
    in_triple = None                                        # the active triple-quote delimiter, or None
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw
        if in_triple:
            if in_triple in line:
                line = line.split(in_triple, 1)[1]
                in_triple = None
            else:
                continue
        # detect a docstring/triple-string opening that doesn't close on the same line
        for delim in ('"""', "'''"):
            if line.count(delim) % 2 == 1:
                line = line.split(delim, 1)[0]
                in_triple = delim
                break
        code = line.split("#", 1)[0]                        # drop line comments
        yield i, code


def main() -> int:
    offenders = []
    for py in sorted(_PKG_DIR.rglob("*.py")):
        rel = py.relative_to(_PKG_DIR).as_posix()
        if rel in _ALLOWLIST or "__pycache__" in rel:
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for lineno, code in _code_lines(text):
            if _IMPORT_RE.search(code) or _SUBPROC_RE.search(code):
                offenders.append(f"  {rel}:{lineno}: {code.strip()}")

    if offenders:
        print("FAIL: dynamic_gs2 has load-bearing references to the old dynamic_gs package:")
        print("\n".join(offenders))
        print(f"\n({len(offenders)} offending line(s). Allowlist: {sorted(_ALLOWLIST)})")
        return 1

    print("test_no_old_package_imports OK  (dynamic_gs2 is import-independent of dynamic_gs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
