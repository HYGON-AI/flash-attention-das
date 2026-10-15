#!/usr/bin/env python3
"""Convert optest .optest-results.json to JUnit XML for GitLab CI.

optest output structure:
  /op_test_output/flashattn/<branch>_<8char_commit>_<timestamp>/
    ├── .optest-results.json
    ├── acc/
    └── *.csv

Logs mirror the same version directory under /op_test_log/flashattn/.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring
import xml.dom.minidom

_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
_FATAL_RE = re.compile(r"^Fatal Python error:")
_SUMMARY_RE = re.compile(r"^=* short test summary info =*$")
_TRACEBACK = "Traceback (most recent call last):"
_MAX_FULL_LINES = 2000
_FATAL_CONTEXT = 100

RESULTS_ROOT = Path("/op_test_output/flashattn")
LOG_ROOT = Path("/op_test_log/flashattn")
_RUNNER_TAG = sys.argv[2] if len(sys.argv) >= 3 else "unknown"
JUNIT_OUTPUT = (
    Path(sys.argv[1]) / f"tests/{_RUNNER_TAG}_flashattn_performance_output.xml"
    if len(sys.argv) > 1
    else Path(f"tests/{_RUNNER_TAG}_flashattn_performance_output.xml")
)



def find_latest_results() -> Path | None:
    """Return the newest .optest-results.json under RESULTS_ROOT."""
    candidates = sorted(RESULTS_ROOT.glob("*/.optest-results.json"))
    return candidates[-1] if candidates else None


def make_error_suite(message: str) -> str:
    testsuite = Element(
        "testsuite", name=f"{_RUNNER_TAG}.flashattn.performance",
        tests="1", failures="1", errors="0", skipped="0",
    )
    tc = SubElement(testsuite, "testcase",
                    classname=f"{_RUNNER_TAG}.flashattn.performance", name="no_results")
    SubElement(tc, "failure", message=message)
    return xml.dom.minidom.parseString(tostring(testsuite)).toprettyxml()


def _read_error_context(log_dir: Path, log_rel: str) -> str:
    """Extract the most relevant error information from a log file.

    - Total lines ≤ _MAX_FULL_LINES → full content.
    - Larger logs → smart extraction (see inline comments).
    ANSI escape codes are stripped for readability in GitLab's JUnit viewer.
    """
    log_file = log_dir / log_rel
    if not log_file.is_file():
        return f"(log file not found: {log_file})"
    try:
        content = log_file.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"(failed to read log: {exc})"

    lines = content.splitlines()

    # ── small enough → return everything ─────────────────────────────
    if len(lines) <= _MAX_FULL_LINES:
        return "\n".join(_ANSI_RE.sub("", line) for line in lines)

    # ── smart extraction for large logs ──────────────────────────────

    # 1. Fatal Python error (SIGSEGV / SIGABRT / …).
    #    Include 100 lines before the marker for debugging context.
    for i, line in enumerate(lines):
        if _FATAL_RE.match(line):
            start = max(0, i - _FATAL_CONTEXT)
            return "\n".join(_ANSI_RE.sub("", l) for l in lines[start:])

    # 2. pytest short test summary (per-failure reasons + final tally).
    for i, line in enumerate(lines):
        if _SUMMARY_RE.match(line):
            return "\n".join(_ANSI_RE.sub("", l) for l in lines[i:])

    # 3. First Python traceback (unhandled exception).
    #    Include 100 lines before for context.
    for i, line in enumerate(lines):
        if line == _TRACEBACK:
            start = max(0, i - _FATAL_CONTEXT)
            return "\n".join(_ANSI_RE.sub("", l) for l in lines[start:])

    # 4. Fallback: last 200 lines.
    return "\n".join(_ANSI_RE.sub("", line) for line in lines[-200:])


def build_junit(data: dict, log_dir: Path) -> str:
    testsuite = Element(
        "testsuite", name=f"{_RUNNER_TAG}.flashattn.performance",
        tests="0", failures="0", errors="0", skipped="0",
    )
    total = 0
    failed = 0
    skipped = 0

    # ── per (performance) tests ──────────────────────────────────────
    per_data = data.get("per", {})
    for test in per_data.get("tests", []):
        total += 1
        name = test["name"]
        duration = test.get("duration_s")
        attrs = {"classname": f"{_RUNNER_TAG}.flashattn.per", "name": name}
        if duration is not None:
            attrs["time"] = str(duration)
        tc = SubElement(testsuite, "testcase", **attrs)
        if test.get("status") == "FAIL":
            failed += 1
            log_rel = test.get("log", "")
            failure = SubElement(
                tc, "failure",
                message=f"{name} failed (log: {log_rel})",
            )
            failure.text = _read_error_context(log_dir, log_rel)

    per_summary = per_data.get("summary", {})
    per_failed = per_summary.get("failed", 0)
    if per_failed > failed:
        failed = per_failed

    # ── acc (accuracy / pytest) tests ─────────────────────────────────
    acc_data = data.get("acc", {})
    for row in acc_data.get("rows", []):
        total += 1
        name = row["name"]
        row_passed = row.get("passed", 0)
        row_failed = row.get("failed", 0)
        row_skipped = row.get("skipped", 0)

        tc = SubElement(
            testsuite, "testcase",
            classname=f"{_RUNNER_TAG}.flashattn.acc",
            name=f"{name} (p:{row_passed} f:{row_failed} s:{row_skipped})",
        )
        if row_failed > 0:
            failed += 1
            failure = SubElement(
                tc, "failure",
                message=f"{name}: {row_failed}/{row.get('total', 0)} failed",
            )
            failure.text = _read_error_context(log_dir, f"acc/{name}.log")
        if row_skipped > 0:
            skipped += 1
            SubElement(tc, "skipped",
                       message=f"{name}: {row_skipped}/{row.get('total', 0)} skipped")

    testsuite.set("tests", str(total))
    testsuite.set("failures", str(failed))
    testsuite.set("skipped", str(skipped))

    return xml.dom.minidom.parseString(tostring(testsuite)).toprettyxml()


def main() -> int:
    results_file = find_latest_results()
    if results_file is None:
        xml_str = make_error_suite("No optest results found under " + str(RESULTS_ROOT))
        JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
        print(f"No optest results found; wrote empty JUnit report to {JUNIT_OUTPUT}")
        return 0

    data = json.loads(results_file.read_text(encoding="utf-8"))
    branch = data.get("branch", "?")
    commit = data.get("commit", "?")[:8]
    version_dir = results_file.parent.name
    log_dir = LOG_ROOT / version_dir
    print(f"Converting optest results: {branch=} commit={commit} {version_dir=}")

    xml_str = build_junit(data, log_dir)

    JUNIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    JUNIT_OUTPUT.write_text(xml_str, encoding="utf-8")
    print(f"JUnit report generated: {JUNIT_OUTPUT}")

    # Return non-zero if any failures so GitLab can fail the job.
    per_summary = data.get("per", {}).get("summary", {})
    acc_summary = data.get("acc", {}).get("summary", {})
    total_failed = per_summary.get("failed", 0) + acc_summary.get("failed", 0)
    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
