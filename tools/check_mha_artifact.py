#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Audit a pure HIP MHA library and its generated/installed headers."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence


FORBIDDEN_TEXT = re.compile(
    r"(?:#\s*include\s*[<\"](?:ATen|c10|torch|pybind11|Python)"
    r"|\b(?:torch|c10|at)::|\bPy(?:Object|_Initialize|_Finalize))",
    re.IGNORECASE,
)
FORBIDDEN_ELF = re.compile(
    r"(?:libtorch|libc10|libpython|torch_python|site-packages|dist-packages)",
    re.IGNORECASE,
)


def run(command: Sequence[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.stdout


def scan_tree(root: Path) -> list[str]:
    problems: list[str] = []
    if not root.is_dir():
        return [f"scan directory does not exist: {root}"]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            contents = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        match = FORBIDDEN_TEXT.search(contents)
        if match:
            problems.append(f"{path}: forbidden source contract: {match.group(0)}")
    return problems


def audit_library(library: Path) -> list[str]:
    if not library.is_file():
        return [f"library does not exist: {library}"]
    dynamic = run(["readelf", "-d", str(library)])
    symbols = run(["nm", "-D", "-C", str(library)])
    problems = []
    for label, output in (("dynamic section", dynamic), ("dynamic symbols", symbols)):
        match = FORBIDDEN_ELF.search(output) or FORBIDDEN_TEXT.search(output)
        if match:
            problems.append(f"{label} contains forbidden dependency: {match.group(0)}")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--scan-dir", type=Path, action="append", default=[])
    args = parser.parse_args(argv)

    try:
        problems = audit_library(args.library)
        for directory in args.scan_dir:
            problems.extend(scan_tree(directory))
    except (OSError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        return 2
    print(f"pure HIP MHA audit OK: {args.library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
