#!/usr/bin/env python
"""Fail if any tracked source/doc file contains non-ASCII characters.

Enforces an ASCII-only source convention: mathematical notation in
docstrings, comments, and strings must use plain text or LaTeX (e.g. "rho",
"<psi|", "->") rather than Unicode glyphs. Ruff's RUF001/2/3 only catch the
*confusable* subset (homoglyphs); this check catches every non-ASCII codepoint.

Usage:
    python scripts/check_ascii.py [FILE ...]

With no arguments, scans all git-tracked ``*.py`` and ``*.md`` files. With
arguments (as passed by pre-commit), scans exactly those files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCANNED_SUFFIXES = {'.py', '.md'}


def tracked_files() -> list[str]:
    out = subprocess.run(
        ['git', 'ls-files', '*.py', '*.md'], capture_output=True, text=True, check=True
    )
    return out.stdout.split()


def main(argv: list[str]) -> int:
    files = argv or tracked_files()
    violations: list[str] = []
    for f in files:
        path = Path(f)
        if path.suffix not in SCANNED_SUFFIXES or not path.is_file():
            continue
        for lineno, line in enumerate(
            path.read_text(encoding='utf-8').splitlines(), start=1
        ):
            for col, ch in enumerate(line, start=1):
                if ord(ch) > 127:
                    violations.append(
                        f'{f}:{lineno}:{col}: non-ASCII {ch!r} (U+{ord(ch):04X})'
                    )
    if violations:
        print('Non-ASCII characters found (use plain text or LaTeX instead):')
        for v in violations:
            print(f'  {v}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
