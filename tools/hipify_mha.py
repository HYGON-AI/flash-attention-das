#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: BSD-3-Clause

"""Discover and generate pure HIP MHA sources without importing PyTorch."""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Sequence


LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
FIXED_SOURCES = (
    ("host", PurePosixPath("flash_c_api.cpp")),
    ("header", PurePosixPath("flash_c_api.h")),
)
HEADER_SUFFIXES = {".h", ".hpp", ".cuh"}
EXTERNAL_QUOTED_INCLUDE_PREFIXES = ("hute/", "hytlass/", "cute/", "cutlass/")


class DiscoveryError(ValueError):
    """Raised when the pure HIP MHA source closure cannot be resolved."""


@dataclasses.dataclass(frozen=True, order=True)
class SourceEntry:
    kind: str
    relative_path: PurePosixPath


class GenerationError(RuntimeError):
    """Raised when hipify or a pure HIP MHA transformation fails."""


MHA_RUNTIME_HEADER = r'''#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace flash_attention {

struct PhiloxState {
    union Payload {
        std::uint64_t value;
        std::int64_t* pointer;
    };

    Payload seed{};
    Payload offset{};
    std::uint32_t offset_intragraph = 0;
    bool captured = false;
};

__host__ __device__ __forceinline__ std::tuple<std::uint64_t, std::uint64_t>
unpack_philox(PhiloxState state)
{
    if(state.captured)
    {
        return {static_cast<std::uint64_t>(*state.seed.pointer),
                static_cast<std::uint64_t>(*state.offset.pointer + state.offset_intragraph)};
    }
    return {state.seed.value, state.offset.value};
}

inline hipDeviceProp_t_v2 const* current_device_properties()
{
    static thread_local hipDeviceProp_t_v2 properties{};
    int device = 0;
    if(hipGetDevice(&device) == hipSuccess)
    {
        (void)hipGetDeviceProperties_v2(&properties, device);
    }
    return &properties;
}

#if defined(__HIPCC__)

template <class Packed>
__device__ __forceinline__ Packed packed_add(Packed lhs, Packed rhs)
{
#if defined(__has_builtin) && __has_builtin(__builtin_hcu_pk_add_f32)
    return __builtin_hcu_pk_add_f32(lhs, rhs);
#else
    return lhs + rhs;
#endif
}

template <class Packed>
__device__ __forceinline__ Packed packed_multiply(Packed lhs, Packed rhs)
{
#if defined(__has_builtin) && __has_builtin(__builtin_hcu_pk_mul_f32)
    return __builtin_hcu_pk_mul_f32(lhs, rhs);
#else
    return lhs * rhs;
#endif
}

template <class Packed>
__device__ __forceinline__ Packed packed_fma(Packed lhs, Packed rhs, Packed addend)
{
#if defined(__has_builtin) && __has_builtin(__builtin_hcu_pk_fma_f32)
    return __builtin_hcu_pk_fma_f32(lhs, rhs, addend);
#else
    return Packed{__builtin_fmaf(lhs.x, rhs.x, addend.x),
                  __builtin_fmaf(lhs.y, rhs.y, addend.y)};
#endif
}

__device__ __forceinline__ auto packed_cvt_bf16_f32(float lhs, float rhs, bool clamp)
{
    (void)clamp;
    return __builtin_hcu_cvt_pk_bf16_f32(0, lhs, 0, rhs, 0);
}

__device__ __forceinline__ auto
packed_cvt_f16_f32(float lhs, float rhs, bool clamp, int rounding)
{
    (void)clamp;
    (void)rounding;
    return __builtin_hcu_cvt_pk_f16_f32(0, lhs, 0, rhs, 0, 0);
}

#endif

template <class... Parts>
[[noreturn]] inline void check_failed(char const* expression, Parts&&... parts)
{
    std::ostringstream message;
    (message << ... << std::forward<Parts>(parts));
    throw std::runtime_error(std::string("FlashAttention check failed: ") + expression +
                             ": " + message.str());
}

} // namespace flash_attention

#define FLASH_ATTENTION_CHECK(condition, ...)                                      \
    do                                                                              \
    {                                                                               \
        if(!(condition))                                                            \
            flash_attention::check_failed(#condition, __VA_ARGS__);                 \
    } while(false)

#define FLASH_ATTENTION_HIP_KERNEL_LAUNCH_CHECK()                                   \
    do                                                                              \
    {                                                                               \
        auto flash_attention_hip_error = hipGetLastError();                         \
        if(flash_attention_hip_error != hipSuccess)                                 \
            flash_attention::check_failed("hipGetLastError",                       \
                                          hipGetErrorString(flash_attention_hip_error)); \
    } while(false)
'''

DTK_MACROS_HEADER = r'''#pragma once

#if defined(USE_ROCM)
#define __DTK_ARCH__ 520
#define DTK_VERSION 11800
#define DTKRT_VERSION 11080
#endif
'''


TORCH_INCLUDE_LINE = re.compile(
    r"^\s*#\s*include\s*<(?:ATen|c10|torch|pybind11|Python)[^>]*>[^\n]*\n?",
    re.MULTILINE,
)
FUSED_ZERO_COPY_DECLARATION = re.compile(
    r"\n\s*at::Tensor\s+fused_zero_copy\s*\([^;]*\);\s*",
    re.MULTILINE | re.DOTALL,
)
DEPENDENCY_COMMENT_LINE = re.compile(
    r"^\s*//.*(?:pytorch|torch|ATen|c10|pybind|Python).*$\n?",
    re.MULTILINE | re.IGNORECASE,
)


def _resolve_local_include(
    source_root: Path, including: PurePosixPath, include_name: str
) -> PurePosixPath | None:
    include_path = PurePosixPath(include_name)
    if include_path.is_absolute() or ".." in include_path.parts:
        raise DiscoveryError(f"local include escapes source root: {including}: {include_name}")
    candidates = (
        including.parent / include_path,
        include_path,
        PurePosixPath("src") / include_path,
    )
    for candidate in candidates:
        if (source_root / candidate).is_file():
            return candidate
    if include_name.startswith(EXTERNAL_QUOTED_INCLUDE_PREFIXES):
        return None
    raise DiscoveryError(f"unresolved local include: {including}: {include_name}")


def discover_sources(source_root: Path) -> list[SourceEntry]:
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise DiscoveryError(f"source root does not exist: {source_root}")

    initial: list[SourceEntry] = []
    for kind, relative_path in FIXED_SOURCES:
        if not (source_root / relative_path).is_file():
            raise DiscoveryError(f"required source does not exist: {relative_path}")
        initial.append(SourceEntry(kind, relative_path))

    kernels = sorted((source_root / "src").glob("flash_fwd*.cu"))
    if not kernels:
        raise DiscoveryError("no forward sources match src/flash_fwd*.cu")
    initial.extend(
        SourceEntry("kernel", PurePosixPath(path.relative_to(source_root).as_posix()))
        for path in kernels
    )

    entries: dict[PurePosixPath, SourceEntry] = {
        entry.relative_path: entry for entry in initial
    }
    pending = deque(entry.relative_path for entry in initial)
    while pending:
        relative_path = pending.popleft()
        text = (source_root / relative_path).read_text(encoding="utf-8")
        for include_name in LOCAL_INCLUDE.findall(text):
            dependency = _resolve_local_include(source_root, relative_path, include_name)
            if dependency is None:
                continue
            if dependency in entries:
                continue
            if dependency.suffix not in HEADER_SUFFIXES:
                raise DiscoveryError(
                    f"unsupported local include type: {relative_path}: {dependency}"
                )
            entries[dependency] = SourceEntry("header", dependency)
            pending.append(dependency)

    kind_order = {"host": 0, "header": 1, "kernel": 2}
    return sorted(entries.values(), key=lambda entry: (kind_order[entry.kind], entry.relative_path))


def render_snapshot(entries: Sequence[SourceEntry]) -> str:
    lines = [
        "# Resolved automatically from csrc/flash_attn_cutlass.",
        "# Generated in the build tree; do not edit or commit.",
    ]
    lines.extend(f"{entry.kind} {entry.relative_path.as_posix()}" for entry in entries)
    return "\n".join(lines) + "\n"


def write_if_changed(path: Path, contents: str) -> bool:
    if path.is_file() and path.read_text(encoding="utf-8") == contents:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            temporary_file.write(contents)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return True


def _generated_relative_path(entry: SourceEntry) -> PurePosixPath:
    path = entry.relative_path
    if entry.kind == "host":
        return path.with_name(f"{path.stem}_hip.cpp")
    if entry.kind == "kernel":
        return path.with_suffix(".hip")
    return path


# hipify-perl treats "<" in <= / >= as the start of a templated kernel launch
# when a nearby kernel uses foo<T><<<...>>>. Rewrite those operators before
# hipify and restore them afterwards. <<= / >>= must be protected first because
# they contain the <= / >= substrings.
_HIPIFY_OPERATOR_PROTECTION = (
    ("<<=", "__HIPIFY_LSHIFT_EQ__"),
    (">>=", "__HIPIFY_RSHIFT_EQ__"),
    ("<=", "__HIPIFY_LE__"),
    (">=", "__HIPIFY_GE__"),
)


def _protect_hipify_operators(contents: str) -> str:
    for source, token in _HIPIFY_OPERATOR_PROTECTION:
        contents = contents.replace(source, token)
    return contents


def _restore_hipify_operators(contents: str) -> str:
    for source, token in reversed(_HIPIFY_OPERATOR_PROTECTION):
        contents = contents.replace(token, source)
    return contents


def _hipify(hipify_perl: Path, input_path: Path) -> str:
    if not hipify_perl.is_file():
        raise GenerationError(f"hipify-perl does not exist: {hipify_perl}")
    protected = _protect_hipify_operators(input_path.read_text(encoding="utf-8"))
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=input_path.suffix or ".cu",
        prefix=f"{input_path.stem}.",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(protected)
        tmp_path = Path(tmp.name)
    try:
        result = subprocess.run(
            [str(hipify_perl), "-hip-kernel-execution-syntax", str(tmp_path)],
            check=False,
            capture_output=True,
            text=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)
    if result.returncode != 0:
        raise GenerationError(
            f"hipify-perl failed for {input_path} with exit code {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return _restore_hipify_operators(result.stdout)


def _matching_parenthesis(contents: str, opening: int) -> int:
    depth = 0
    quote = ""
    escaped = False
    index = opening
    while index < len(contents):
        character = contents[index]
        following = contents[index + 1] if index + 1 < len(contents) else ""
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
        elif character in ('"', "'"):
            quote = character
        elif character == "/" and following == "/":
            newline = contents.find("\n", index + 2)
            index = len(contents) if newline < 0 else newline
            continue
        elif character == "/" and following == "*":
            closing = contents.find("*/", index + 2)
            if closing < 0:
                raise GenerationError("unterminated comment in generated HIP source")
            index = closing + 2
            continue
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise GenerationError("unterminated hipLaunchKernelGGL call")


def _split_call_arguments(arguments: str) -> list[str]:
    result: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    quote = ""
    escaped = False
    for index, character in enumerate(arguments):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = ""
            continue
        if character in ('"', "'"):
            quote = character
        elif character in "([{":
            stack.append(character)
        elif character in ")]}" and stack:
            if stack[-1] != pairs[character]:
                raise GenerationError("unbalanced delimiter in hipLaunchKernelGGL call")
            stack.pop()
        elif character == "," and not stack:
            result.append(arguments[start:index].strip())
            start = index + 1
    if quote or stack:
        raise GenerationError("unbalanced argument in hipLaunchKernelGGL call")
    result.append(arguments[start:].strip())
    return result


def _canonicalize_hip_launches(contents: str) -> str:
    marker = "hipLaunchKernelGGL("
    output: list[str] = []
    cursor = 0
    while True:
        start = contents.find(marker, cursor)
        if start < 0:
            output.append(contents[cursor:])
            break
        opening = start + len(marker) - 1
        closing = _matching_parenthesis(contents, opening)
        arguments = _split_call_arguments(contents[opening + 1 : closing])
        if len(arguments) < 5:
            raise GenerationError("hipLaunchKernelGGL requires at least five arguments")
        kernel = arguments[0]
        if kernel.startswith("HIP_KERNEL_NAME(") and kernel.endswith(")"):
            kernel = kernel[len("HIP_KERNEL_NAME(") : -1].strip()
        if not (kernel.startswith("(") and kernel.endswith(")")):
            kernel = f"({kernel})"
        grid = arguments[1]
        block = arguments[2]
        if not grid.startswith("dim3("):
            grid = f"dim3({grid})"
        if not block.startswith("dim3("):
            block = f"dim3({block})"
        canonical = ", ".join([kernel, grid, block, *arguments[3:]])
        output.extend((contents[cursor:start], marker, canonical, ")"))
        cursor = closing + 1
    transformed = "".join(output)
    if "<<<" in transformed:
        raise GenerationError("CUDA kernel launch syntax remains after hipify")
    return transformed


def _inject_hip_runtime_for_device_logic(contents: str) -> str:
    has_runtime = re.search(
        r'#\s*include\s*[<"]hip/hip_runtime(?:_api)?\.h[>"]', contents
    )
    has_device_logic = (
        "hipLaunchKernelGGL" in contents
        or "__global__" in contents
        or "__shared__" in contents
        or re.search(r"\b__syncthreads\w*\(", contents) is not None
    )
    if has_device_logic and not has_runtime:
        return '#include "hip/hip_runtime.h"\n' + contents
    return contents


def _transform_mha_source(contents: str, entry: SourceEntry) -> str:
    contents = _canonicalize_hip_launches(contents)
    contents = contents.replace("__CUDA_ARCH__", "__DTK_ARCH__")
    contents = contents.replace("CUDART_VERSION", "DTKRT_VERSION")
    contents = contents.replace("CUDA_VERSION", "DTK_VERSION")
    contents = TORCH_INCLUDE_LINE.sub("", contents)
    contents = DEPENDENCY_COMMENT_LINE.sub("", contents)
    contents = contents.replace("#include <cuda_bf16.h>", "#include <hip/hip_bf16.h>")
    contents = re.sub(r"\bhipDeviceProp_t\b", "hipDeviceProp_t_v2", contents)
    contents = re.sub(
        r"\bhipGetDeviceProperties\b", "hipGetDeviceProperties_v2", contents
    )
    contents = contents.replace("at::PhiloxCudaState", "flash_attention::PhiloxState")
    contents = contents.replace(
        "at::cuda::philox::unpack", "flash_attention::unpack_philox"
    )
    contents = contents.replace(
        "at::cuda::getCurrentDeviceProperties", "flash_attention::current_device_properties"
    )
    contents = contents.replace("TORCH_CHECK(", "FLASH_ATTENTION_CHECK(")
    contents = contents.replace(
        "C10_CUDA_KERNEL_LAUNCH_CHECK()", "FLASH_ATTENTION_HIP_KERNEL_LAUNCH_CHECK()"
    )
    contents = contents.replace(
        "C10_HIP_KERNEL_LAUNCH_CHECK()", "FLASH_ATTENTION_HIP_KERNEL_LAUNCH_CHECK()"
    )
    contents = contents.replace(
        "__builtin_hcu_pk_add_f32", "flash_attention::packed_add"
    )
    contents = contents.replace(
        "__builtin_hcu_pk_mul_f32", "flash_attention::packed_multiply"
    )
    contents = contents.replace(
        "__builtin_hcu_pk_fma_f32", "flash_attention::packed_fma"
    )
    contents = contents.replace(
        "__builtin_hcu_cvt_pk_bf16_f32(",
        "flash_attention::packed_cvt_bf16_f32(",
    )
    contents = contents.replace(
        "__builtin_hcu_cvt_pk_f16_f32(",
        "flash_attention::packed_cvt_f16_f32(",
    )
    contents = contents.replace("Pytorch", "the pure HIP MHA runtime")
    contents = contents.replace("PyTorch", "the pure HIP MHA runtime")
    if entry.relative_path == PurePosixPath("flash_c_api.h"):
        contents = contents.replace(
            '#include "flash_attnmask.h"', '#include "src/flash_attnmask.h"'
        )
    if entry.relative_path == PurePosixPath("src/flash_attnmask.h"):
        contents = FUSED_ZERO_COPY_DECLARATION.sub("\n", contents)
    if entry.relative_path == PurePosixPath("src/flash.h"):
        marker = "#pragma once"
        if marker not in contents:
            raise GenerationError("generated src/flash.h has no #pragma once insertion point")
        contents = contents.replace(
            marker, f'{marker}\n\n#include "flash_mha_runtime.h"', 1
        )
    contents = _inject_hip_runtime_for_device_logic(contents)
    return '#include "src/flash_dtk_macros.h"\n' + contents


def generate(
    source_root: Path,
    output_root: Path,
    entries: Sequence[SourceEntry],
    hipify_perl: Path,
    jobs: int,
) -> tuple[int, int]:
    source_root = source_root.resolve()

    def generate_one(entry: SourceEntry) -> bool:
        try:
            generated = _hipify(hipify_perl, source_root / entry.relative_path)
            generated = _transform_mha_source(generated, entry)
        except GenerationError as error:
            raise GenerationError(f"{entry.relative_path}: {error}") from error
        output_path = output_root / _generated_relative_path(entry)
        return write_if_changed(output_path, generated)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        changed = sum(executor.map(generate_one, entries))
    changed += write_if_changed(
        output_root / "src" / "flash_mha_runtime.h",
        MHA_RUNTIME_HEADER,
    )
    changed += write_if_changed(
        output_root / "src" / "flash_dtk_macros.h",
        DTK_MACROS_HEADER,
    )
    return len(entries) + 2, changed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--check-sources", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--hipify-perl", type=Path)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.check_sources and not args.generate:
        print("error: no action selected; use --check-sources or --generate", file=sys.stderr)
        return 2
    try:
        entries = discover_sources(args.source_root)
        snapshot = args.snapshot
        if args.generate:
            if args.output_root is None:
                raise GenerationError("--generate requires --output-root")
            if args.hipify_perl is None:
                raise GenerationError("--generate requires --hipify-perl")
            if args.jobs < 1:
                raise GenerationError("--jobs must be at least 1")
            if snapshot is None:
                snapshot = args.output_root / "resolved_sources.txt"
            generated_count, changed_count = generate(
                args.source_root, args.output_root, entries, args.hipify_perl, args.jobs
            )
        if snapshot:
            write_if_changed(snapshot, render_snapshot(entries))
    except (DiscoveryError, GenerationError, OSError, UnicodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    kernel_count = sum(entry.kind == "kernel" for entry in entries)
    header_count = sum(entry.kind == "header" for entry in entries)
    print(
        f"source discovery OK: {len(entries)} entries, "
        f"{header_count} header(s), {kernel_count} kernel source(s)"
    )
    if args.generate:
        print(f"generation OK: {generated_count} output(s), {changed_count} changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
