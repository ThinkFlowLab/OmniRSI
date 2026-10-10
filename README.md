# OmniRSI

面向推理系统的 **CLI 实验工作台与调优知识库**。将一次优化的源码身份、参数、基线、候选、质量验证与原始结果放在同一条证据链中。

初版实现 reviewed plan 的第一个可运行切片：用户提供实际 benchmark 和 validation 命令，OmniRSI 负责重复执行、记录、比较和报告。所有运行配置使用 **CLI args**，没有 YAML 任务输入；控制包无必需第三方运行依赖，也不导入 torch 或推理引擎。

## 已实现

- `doctor`：检查本机准备好的 Git checkout、Python 包版本与后端工具；CUDA 记录设备、驱动和拓扑。工具存在不等于模型/功能已兼容。
- `plan` / `run --dry-run`：显示解析后的 RunSpec 和环境检查，不启动实验。
- `run`：交替 AB/BA 重复执行明确的 baseline/candidate argv，在全局实验期限内保存结果、日志、源码指纹与质量命令证据。
- `status` / `report`：读取运行状态、重新生成可独立浏览的 HTML 报告。
- `knowledge search/show`：检索包内的来源固定知识；按仓库、后端和场景过滤。
- `context`：导出带来源、适用条件、验证状态和限制的 Markdown，供 Codex / 三方 Agent 人工接续。

初版不自动生成或应用代码补丁、不启动 Agent、不部署模型、不激活生产服务，也不实现控制端 SSH 编排、设备租约或崩溃恢复。**在准备好的 remote 机器上安装包后使用 local executor。** `--agent` 仅记录交接对象，不调用该 Agent。

## Quickstart

需要 Python 3.10+、Git，以及可运行目标仓库的环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
omnirsi --help
omnirsi --version
```

PowerShell 激活命令为 `.\.venv\Scripts\Activate.ps1`。以下命令在已 checkout 此 PR 的 OmniRSI 根目录执行。

### CPU 合约 smoke：不是性能或质量基准

```bash
omnirsi doctor --repo . --repo-type vllm_omni --backend cpu

omnirsi run \
  --repo . --repo-type vllm_omni --backend cpu \
  --scenario smoke.synthetic \
  --baseline-command 'python examples/synthetic_benchmark.py --value 100 --output {result}' \
  --candidate-command 'python examples/synthetic_benchmark.py --value 80 --output {result}' \
  --validation-command 'python examples/synthetic_validation.py' \
  --metric-key metrics.latency_ms --metric-unit ms \
  --comparison-scope synthetic_fixture --direction minimize \
  --repetitions 3 --min-improvement-pct 5 --max-minutes 2 \
  --output-dir runs
```

此 fixture 写入预设的 100 / 80 数值，预期 `PASS` 和 20% 数值差；它只证明执行、证据和判定链路工作，没有测量模型加速。实际命令需要替换为目标仓库的 benchmark 与质量检查。

每个 trial 的 `{result}` 被替换为独立 JSON 路径，也通过 `OMNIRSI_RESULT_PATH` 提供。命令以 argv 执行，`shell=False`；不会展开 shell 管道、环境变量或重定向。需要这些操作时使用自己审阅的脚本。

```bash
omnirsi status --run-id RUN_ID --output-dir runs
omnirsi report --run-id RUN_ID --output-dir runs
```

详见 [完整 Quickstart](docs/quickstart.md)、[远程 Test plan](docs/test-plan.md)。

[B300 四卡 / Qwen-Image-2.1-Turbo 性能计划](docs/qwen21-turbo-b300-test-plan.md)是当前 campaign：固定最新 main 和公开模型 revision，720p 档位、完整 8-step schedule、原生 RGBA PNG；先 profile，再选择质量合格的四卡并行配置并分层优化，以五组 AB/BA 验证至少 20% E2E latency 下降。性能与精度结论以实测报告为准。

[本地实测报告](docs/qwen21-turbo-b300-results.md)：五组 AB/BA 为 385.07→335.42 ms，降低 **12.89%**，未达到 20% 目标；所有最终/held-out RGBA 图像的解码像素一致。报告保留被拒绝的迭代和基础设施失败，不把 GPU 计划当作性能证据。

此前 [LTX-2.5 计划](docs/ltx25-b300-test-plan.md)及[授权阻塞记录](docs/ltx25-b300-local-status.md)保留为历史，当前任务不再依赖该模型授权。

## 知识库

```bash
omnirsi knowledge search 'MiniMax-H3' --repo-type vllm_omni --backend cuda
omnirsi knowledge search --repo-type vllm_omni --backend ascend --scenario diffusion.video_generation --json
omnirsi knowledge show vllm_omni.common.diffusion.measurement_and_profiling
omnirsi context 'MiniMax-H3' --repo-type vllm_omni --backend ascend --output runs/h3-context.md
```

检索按关键词、repo/backend/scenario 过滤；具体设备、模型、软件和负载边界保留在记录中，初版需要人工核对。`common` 方法可与具体 diffusion 场景一起检索；`diffusion.t2i/t2v` 是图像/视频场景别名。

初始导入包括两份指定技能和 8 个 MiniMax-H3 PR：

| 条目 | 核心经验 |
| --- | --- |
| diffusion-perf-opt | 固定三段复现命令；性能测量与 profiling 分开；一次实验控制变量 |
| production-add-diffusion-model | loading 支持与生产 readiness 分开；模型/任务/后端逐项验证 |
| [#5990](https://github.com/vllm-project/vllm-omni/pull/5990) | Q/K norm 与 RoPE 融合的形状、平台和输出边界 |
| [#6281](https://github.com/vllm-project/vllm-omni/pull/6281) | modulation 融合的舍入差异，以及非 CUDA 平台 guard |
| [#5783](https://github.com/vllm-project/vllm-omni/pull/5783) | AdaLN cache 的容量收益与速度收益分开评估 |
| [#5952](https://github.com/vllm-project/vllm-omni/pull/5952) | Ref2VA TeaCache 的任务校准、步数和质量要求 |
| [#7024](https://github.com/vllm-project/vllm-omni/pull/7024) | VAE stacked tiling 的调用次数与显存取舍 |
| [#7988](https://github.com/vllm-project/vllm-omni/pull/7988) | MXFP8 是近似部署路线，需要质量和加载显存验证 |
| [#7514](https://github.com/vllm-project/vllm-omni/pull/7514) | Ascend SP 通信分桶与尚未完成的整体质量等价验证 |
| [#8072](https://github.com/vllm-project/vllm-omni/pull/8072) | reference KV 缓存位置、并行 ownership 与退化反例 |

来源技能固定于 `ee81f33001922f8d91dd09da0a57ee1531a8df82`；各 PR 分别固定导入时的 head 和观察状态。**全部条目为 `imported`、`locally_verified=false`。** merged/open、上游报告、实际测量硬件和未知质量边界均单独保存。没有将 ROCm 的其他模型经验冒充 H3 证据。

[知识索引、来源和使用建议](src/omnirsi/data/repo_packs/vllm_omni/knowledge/README.md)。

## 目录结构

```text
src/omnirsi/
  arguments.py / cli.py / contracts.py     CLI 与冻结契约
  inspection.py                           环境与仓库检查
  execution.py                            本机显式命令执行
  evaluation.py / reporting.py            证据判定与 HTML
  catalog.py                              只读知识检索
  data/
    repo_packs/{vllm,vllm_omni,afd_plugin,vllm_rlt}/
    platform_packs/{cuda,rocm,ascend}/
    repo_packs/vllm_omni/knowledge/experiences/
      common/diffusion/...
      cuda/diffusion/...
      ascend/diffusion/...
examples/                                 CPU smoke fixture
tests/                                    运行、CLI 与知识契约
docs/                                     使用、架构与远程 Test plan
runs/                                     本地生成，默认忽略
```

知识和描述包以 `src/omnirsi/data/` 为唯一权威位置，通过 package data 进入 wheel，避免安装后依赖源码 checkout 或出现两份经验副本。四类仓库描述当前只声明外部命令入口，并未实现全套模型 launcher；三类加速器兼容矩阵仍为 unverified。

## 组件与流程

[组件结构、完整流程图与实际初版边界](docs/architecture.md)。用户给定候选命令相当于为 Phase 3 提供待验证实现；初版执行测量、质量验证、判定、报告，不把计划中未实现的阶段写成已经完成。

## 判定与复现边界

- `PASS`：请求的 trial 完整、命令成功、指标有效、源码指纹不变、显式 validation 成功，且正向中位数收益达到阈值。
- `FAIL`：命令失败、超时、质量命令失败，或完整测量没有达到收益要求。
- `INCONCLUSIVE`：质量命令缺失、JSON/指标无效、源码不可确定/发生变化、取消或证据不足。
- 中位数与范围是描述性统计，不证明显著性。质量命令退出 0 仅覆盖其实际检查范围，不能自动证明模型等价或服务 readiness。
- 指标单位、测量范围、模型/负载身份由调用者声明。外部 server 的实际版本、负载一致性、失败请求和 GPU 资源公平性需要另行核对。
- Git 记录是 HEAD、dirty diff/untracked 哈希与前后指纹，不归档完整脏源码。复现时使用 committed baseline/candidate，或另行保留补丁和未追踪文件。
- `--device-ids` 是元数据，不设置可见设备或占用租约；先自行分配资源并设置后端可见性。
- `--python` 仅检查那个解释器的包；benchmark/validation 使用命令中写明的解释器。`--candidate-repo` 仅改变 cwd，不切换已经安装的包；候选需要独立环境或明确的导入路径。
- 期限限制测量与质量命令的启动和等待；源码检查、清理与最终报告可能额外耗时。初版预算上限七天。Ctrl-C 保存状态与证据，但不支持跨控制程序重启 resume。

## 验证与贡献

```bash
python -m unittest discover -s tests -v
python -m pip wheel --no-deps . -w dist
```

本地 CPU 合约验证和实际 GPU/NPU 测量分别记录。远程 CUDA / ROCm / Ascend、模型质量、Linux 进程组与生产长稳由用户按 Test plan 验证。开发遵循 plan → code → test → review；所有提交使用 `git commit -s`。
