#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Validate the HG split/single wheel layout and symbol contract."""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_CHECKER = ROOT / "tools" / "check_mha_artifact.py"

PUBLIC_SYMBOLS = (
    "_Z11run_mha_fwdR16Flash_fwd_paramsP12ihipStream_tb",
    "_Z11run_mha_bwdR16Flash_bwd_paramsP12ihipStream_tb",
    "_Z19run_mha_fwd_kvcacheR16Flash_fwd_paramsP12ihipStream_tb",
)
PRIVATE_SYMBOLS = (
    "_Z20flash_hg_run_mha_fwdR16Flash_fwd_paramsP12ihipStream_tb",
    "_Z20flash_hg_run_mha_bwdR16Flash_bwd_paramsP12ihipStream_tb",
    "_Z28flash_hg_run_mha_fwd_kvcacheR16Flash_fwd_paramsP12ihipStream_tb",
)


def run(cmd):
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    return proc.stdout


def symbol_types(so_path):
    out = run(["nm", "-D", str(so_path)])
    table = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            table[parts[-1]] = parts[-2]
    return table


def has_symbolic(so_path):
    out = run(["readelf", "-d", str(so_path)])
    return "SYMBOLIC" in out


def has_needed(so_path, name):
    out = run(["readelf", "-d", str(so_path)])
    return name in out


def has_runpath_origin(so_path):
    out = run(["readelf", "-d", str(so_path)])
    return "$ORIGIN" in out


def extract_libs(wheel):
    tmp = Path(tempfile.mkdtemp(prefix="hg-wheel-check-"))
    with zipfile.ZipFile(wheel) as zf:
        zf.extractall(tmp)
    lib_dir = tmp / "flash_attn" / "lib"
    if not lib_dir.is_dir():
        raise AssertionError(f"missing flash_attn/lib in {wheel}")
    return tmp, lib_dir


def check_single(lib_dir):
    libs = sorted(p.name for p in lib_dir.iterdir() if p.is_file())
    if libs != ["libflash_attention.so"]:
        raise AssertionError(
            f"single-file wheel must contain only libflash_attention.so, got {libs}"
        )
    adapter = lib_dir / "libflash_attention.so"
    types = symbol_types(adapter)
    missing_public = [s for s in PUBLIC_SYMBOLS if types.get(s) != "T"]
    if missing_public:
        raise AssertionError(f"single-file missing public T symbols: {missing_public}")
    leaked_private = [s for s in PRIVATE_SYMBOLS if s in types]
    if leaked_private:
        raise AssertionError(
            f"single-file must not export private bridge symbols: {leaked_private}"
        )
    print("single-file layout/symbol check OK")


def check_split(lib_dir):
    core = lib_dir / "libflash_attention_c.so"
    adapter = lib_dir / "libflash_attention.so"
    if not core.is_file() or not adapter.is_file():
        raise AssertionError(
            f"split wheel must contain core + adapter, got {sorted(p.name for p in lib_dir.iterdir())}"
        )

    audit = run([sys.executable, str(ARTIFACT_CHECKER), "--library", str(core)])
    if "pure HIP MHA audit OK" not in audit:
        raise AssertionError(f"core torch-free audit failed:\n{audit}")

    adapter_types = symbol_types(adapter)
    core_types = symbol_types(core)
    bad_public = [s for s in PUBLIC_SYMBOLS if adapter_types.get(s) != "T"]
    if bad_public:
        raise AssertionError(f"adapter public symbols are not T: {bad_public}")
    bad_private = [s for s in PRIVATE_SYMBOLS if adapter_types.get(s) != "U"]
    if bad_private:
        raise AssertionError(f"adapter private bridge symbols are not U: {bad_private}")
    bad_core_private = [s for s in PRIVATE_SYMBOLS if core_types.get(s) != "T"]
    if bad_core_private:
        raise AssertionError(f"core private bridge symbols are not T: {bad_core_private}")

    if not has_symbolic(core) or not has_symbolic(adapter):
        raise AssertionError("core and adapter must both carry FLAGS SYMBOLIC")
    if not has_needed(adapter, "libflash_attention_c.so"):
        raise AssertionError("adapter does not declare NEEDED libflash_attention_c.so")
    if not has_runpath_origin(adapter):
        raise AssertionError("adapter does not carry RUNPATH $ORIGIN")
    print("split layout/symbol/binding/audit check OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True)
    parser.add_argument("--expected-mode", choices=("single", "split"), required=True)
    args = parser.parse_args()

    wheel = Path(args.wheel)
    if not wheel.is_file():
        raise SystemExit(f"wheel not found: {wheel}")

    tmp, lib_dir = extract_libs(wheel)
    try:
        if args.expected_mode == "single":
            check_single(lib_dir)
        else:
            check_split(lib_dir)
    finally:
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
