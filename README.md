# flash-attention-das

[English](#english) | [简体中文](#zh-cn)

<a id="english"></a>

## English

Built on flash-attention, `flash-attention-das` provides attention kernels, Python APIs and HIP build tools for Hygon HCU. It includes implementations for MHA/MQA/GQA, variable-length sequences, KV cache, paged attention, sparse attention and FP8.

### Upstream and License

| Item | Reference |
| --- | --- |
| Upstream project | flash-attention |
| Repository | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention.git) |
| Upstream baseline tag | `v2.6.1` |
| Upstream baseline commit | `7551202cb2dd245432bc878447e19015c0af3c22` |
| License | [BSD-3-Clause](LICENSE) |
| Upstream README snapshot | [README_ORIGIN.md](README_ORIGIN.md) |

### Hygon Contributions

Modified by Hygon Information Technology Co., Ltd.

Copyright (c) 2026 Hygon Information Technology Co., Ltd.

- Port FlashAttention to HCU and the HIP/AICC toolchain.
- Adapt and optimize attention operators for HCU, including FP8, paged attention and sparse attention.
- Extend the Python APIs and integrate HCU builds, packaging, tests and benchmarks.

The HYGON copyright statement applies only to its original contributions and does not replace or override the copyright statements of Tri Dao or other upstream authors. The original flash-attention copyright and BSD license text remain in [LICENSE](LICENSE), with contributor attribution in [AUTHORS](AUTHORS) and source headers. See [NOTICE](NOTICE) for the repository contribution statement, [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party copyrights, sources and licenses, and [LICENSES](LICENSES/) for third-party license texts.

### Build and Install

#### Python Extension

Requires Linux on Hygon HCU, Python 3.8+, compatible HCU builds of PyTorch and Triton, and DTK with `aicc`, `hipcc` and the device libraries for the build targets. Check out the `csrc/cutlass` submodule and load the DTK environment, then run from the repository root.

```bash
python -m pip install setuptools wheel packaging psutil ninja einops
python -m pip install -v --no-build-isolation --no-deps .
python -m pip show flash-attn
```

The main extension targets `gfx938`, `gfx936`, `gfx928` and `gfx92a`. The Hygon backend is built by default for `gfx938`, `gfx936` and `gfx92a`. Use `FLASH_BUILD_HG=0` to disable the Hygon backend and `MAX_JOBS` to limit build parallelism. [Build configuration](setup.py) · [Wheel build script](ci_script/build_wheel_and_manylinux.sh)

#### Standalone HIP MHA Library

Requires Python 3.8+, CMake 3.15+, Ninja or Make, and a DTK installation with a HIP compiler, the HIP CMake package and `hipify-perl`.

[compile.sh](compile.sh) loads `${ROCM_PATH:-/opt/dtk}/env.sh` and prefers `aicc`. The default targets are `gfx938`, `gfx936`, `gfx928` and `gfx92a`. The default install directory is `build-mha-hip/install/`, with install rules for `libmulti-head-attention.so`, public headers and the `FlashAttention` CMake package. Run `bash compile.sh --help` for options such as `MHA_GPU_ARCHS` and `MAX_JOBS`.

### API Guide and Examples

The [FlashAttention API guide](docs/flash_attention_api.md) (in Chinese) explains the [attention computation](docs/flash_attention_api.md#attention算子介绍), lists the [APIs and their input constraints](docs/flash_attention_api.md#接口列表), and provides examples for [fixed-length attention](docs/flash_attention_api.md#flash_attn_func) and [variable-length GQA](docs/flash_attention_api.md#flash_attn_varlen_func).

For sparse GQA, sparse prefill and decode, and MLA compatibility APIs, see the [Python exports](flash_attn/__init__.py) and [interface implementations](flash_attn/flash_attn_interface.py). A [Triton interface](flash_attn/flash_attn_triton_interface.py) is also available.

HCU PyTorch uses the `torch.cuda` API and the `device="cuda"` device identifier. Before running the examples, make sure your Python environment can load the compiled package.

### Tests and Benchmarks

For a quick check of forward attention, run the examples from the repository root in an HCU environment with the Python extension installed.

#### Attention Mask Correctness Test

The test compares FP16 fixed-length attention with an all-true mask against a PyTorch reference.

```bash
python -m pip install pytest
python -m pytest -q tests/test_flash_attn_attnmask.py::test_attnmask_all_ones
```

#### Attention Mask Forward Benchmark

The benchmark reports FP16 causal attention latency and throughput with and without an explicit mask.

```bash
python benchmarks/benchmark_attnmask.py --sizes 1,256 --nheads 8 --num-heads-kv 2 \
    --headdim 128 --no-backward --no-both-causal
```

Additional correctness tests and benchmark scripts are available in [tests/](tests/) and [benchmarks/](benchmarks/), respectively.

### Code Layout

| Path | Contents |
| --- | --- |
| [flash_attn/](flash_attn/) | Python APIs, backend dispatch and Triton implementations |
| [csrc/flash_attn_cutlass/](csrc/flash_attn_cutlass/) | Main extension bindings and HYTLASS-based attention kernels |
| [csrc/flash_attn_hipc/](csrc/flash_attn_hipc/) | Hygon backend HIP kernels |
| [csrc/cutlass/](csrc/cutlass/) | HYTLASS/HuTe submodule |
| [setup.py](setup.py) | Python extension build and packaging |
| [compile.sh](compile.sh), [CMakeLists.txt](CMakeLists.txt), [cmake/](cmake/) | Standalone HIP MHA source generation, build and installation |
| [tests/](tests/) | Correctness tests |
| [benchmarks/](benchmarks/) | Performance benchmarks |
| [docs/](docs/) | API documentation |

### Acknowledgements and References

Thanks to the authors of FlashAttention and the projects listed in the third-party notices.

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135), Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra and Christopher Ré, 2022.
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf), Tri Dao, 2023.

Retain the applicable copyright, license and paper attribution for the implementations you use.

---

<a id="zh-cn"></a>

## 简体中文

`flash-attention-das` 基于 flash-attention，面向 Hygon HCU 提供注意力计算内核、Python 接口和 HIP 构建工具，包含 MHA/MQA/GQA、变长序列、KV cache、分页注意力、稀疏注意力及 FP8 相关实现。

### 上游来源与许可证

| 项目 | 信息 |
| --- | --- |
| 上游项目 | flash-attention |
| 上游仓库 | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention.git) |
| 上游基线 Tag | `v2.6.1` |
| 上游基线 Commit | `7551202cb2dd245432bc878447e19015c0af3c22` |
| 许可证 | [BSD-3-Clause](LICENSE) |
| 上游 README 备份 | [README_ORIGIN.md](README_ORIGIN.md) |

### Hygon 贡献

Modified by Hygon Information Technology Co., Ltd.

Copyright (c) 2026 Hygon Information Technology Co., Ltd.

Hygon 原创贡献包括：

- 将 FlashAttention 适配到 HCU 平台和 HIP/AICC 工具链。
- 适配和优化面向 HCU 的注意力算子，包括 FP8、分页和稀疏注意力路径。
- 扩展 Python 接口，集成面向 HCU 的构建、打包、测试和性能基准。

HYGON 版权声明仅适用于其原创贡献，不替代或覆盖 Tri Dao 及其他上游作者的版权声明。flash-attention 的原版权和 BSD 许可文字保留在 [LICENSE](LICENSE) 中，作者信息及源文件声明见 [AUTHORS](AUTHORS) 和各文件头。仓库贡献说明见 [NOTICE](NOTICE)，第三方版权、来源和许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)，第三方许可全文保存在 [LICENSES](LICENSES/) 中。

### 构建与安装

#### Python 扩展

需要配备 Hygon HCU 的 Linux 环境、Python 3.8+、匹配的 HCU 版 PyTorch 和 Triton，以及包含 `aicc`、`hipcc` 和各编译目标设备库的 DTK 工具链。检出 `csrc/cutlass` 子模块并加载 DTK 环境后，在仓库根目录执行。

```bash
python -m pip install setuptools wheel packaging psutil ninja einops
python -m pip install -v --no-build-isolation --no-deps .
python -m pip show flash-attn
```

主扩展的编译目标为 `gfx938`、`gfx936`、`gfx928` 和 `gfx92a`。Hygon 后端默认构建，目标为 `gfx938`、`gfx936` 和 `gfx92a`。可用 `FLASH_BUILD_HG=0` 关闭 Hygon 后端，用 `MAX_JOBS` 限制并行构建数。 [构建配置](setup.py) · [Wheel 构建脚本](ci_script/build_wheel_and_manylinux.sh)

#### 独立 HIP MHA 库

需要 Python 3.8+、CMake 3.15+、Ninja 或 Make，以及带有 HIP 编译器、HIP CMake 包和 `hipify-perl` 的 DTK 环境。

[compile.sh](compile.sh) 从 `${ROCM_PATH:-/opt/dtk}/env.sh` 加载环境，优先使用 `aicc`。默认目标为 `gfx938`、`gfx936`、`gfx928` 和 `gfx92a`。默认安装目录为 `build-mha-hip/install/`，安装规则覆盖 `libmulti-head-attention.so`、公开头文件和 `FlashAttention` CMake 包。运行 `bash compile.sh --help` 查看 `MHA_GPU_ARCHS`、`MAX_JOBS` 等选项。

### API 与示例

[FlashAttention API 文档](docs/flash_attention_api.md)介绍了[算子原理与计算流程](docs/flash_attention_api.md#attention算子介绍)、[接口列表与输入约束](docs/flash_attention_api.md#接口列表)，并提供[定长注意力](docs/flash_attention_api.md#flash_attn_func)和[变长 GQA 示例](docs/flash_attention_api.md#flash_attn_varlen_func)的调用示例。

稀疏 GQA、稀疏 prefill/decode 和 MLA 兼容接口可查阅 [Python 导出列表](flash_attn/__init__.py)及[接口实现](flash_attn/flash_attn_interface.py)。仓库也提供 [Triton 接口](flash_attn/flash_attn_triton_interface.py)。

HCU 版 PyTorch 沿用 `torch.cuda` 接口与 `device="cuda"` 标识。运行示例前，请确保当前 Python 环境可以加载已编译的包。

### 测试与基准

在已安装 Python 扩展的 HCU 环境中，从仓库根目录运行以下示例，快速验证注意力前向计算。

#### 注意力掩码正确性测试

测试使用全真掩码，将 FP16 定长注意力的前向结果与 PyTorch 参考结果比较。

```bash
python -m pip install pytest
python -m pytest -q tests/test_flash_attn_attnmask.py::test_attnmask_all_ones
```

#### 注意力掩码前向基准

基准输出 FP16 因果注意力在有显式掩码和无显式掩码两种情况下的延迟和吞吐量。

```bash
python benchmarks/benchmark_attnmask.py --sizes 1,256 --nheads 8 --num-heads-kv 2 \
    --headdim 128 --no-backward --no-both-causal
```

更多正确性测试和基准脚本分别见 [tests/](tests/) 和 [benchmarks/](benchmarks/)。

### 代码结构

| 路径 | 内容 |
| --- | --- |
| [flash_attn/](flash_attn/) | Python 接口、后端分派和 Triton 实现 |
| [csrc/flash_attn_cutlass/](csrc/flash_attn_cutlass/) | 主扩展绑定和基于 HYTLASS 的注意力内核 |
| [csrc/flash_attn_hipc/](csrc/flash_attn_hipc/) | Hygon 后端的 HIP 内核 |
| [csrc/cutlass/](csrc/cutlass/) | HYTLASS/HuTe 子模块 |
| [setup.py](setup.py) | Python 扩展构建与打包 |
| [compile.sh](compile.sh)、[CMakeLists.txt](CMakeLists.txt)、[cmake/](cmake/) | 独立 HIP MHA 源码生成、构建和安装 |
| [tests/](tests/) | 正确性测试 |
| [benchmarks/](benchmarks/) | 性能基准 |
| [docs/](docs/) | API 文档 |

### 致谢与引用

感谢 FlashAttention 及第三方清单中各项目的作者。

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)，Tri Dao、Daniel Y. Fu、Stefano Ermon、Atri Rudra、Christopher Ré，2022。
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://tridao.me/publications/flash2/flash2.pdf)，Tri Dao，2023。

使用相关实现时，请保留其适用的版权、许可和论文引用信息。
