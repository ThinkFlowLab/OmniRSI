# Remote Codex Test plan：B300 × 4 / LTX-2.5

这是待在 remote 执行的计划，**没有声称已经找到最佳配置或获得 30% 加速**。所有运行配置通过 CLI args；JSON 是派生结果、引用与冻结证据。

## 目标与固定协议

先在所列的四 worker 搜索空间中找出质量通过的最低延迟配置，再固定该配置，以新的 AB/BA 复测为基线，优化代码使：

`median(candidate trial mean E2E ms) <= 0.70 × median(best-config baseline trial mean E2E ms)`

这定义为端到端延迟下降至少 30%，等价约 1.43× speedup。它不是相对最初单卡、默认配置或单 kernel 的收益。每 trial 对三个固定请求取均值，最终对五个 trial 均值取中位数；不混用 pooled request median 或生产吞吐。

| 条件 | 本主计划 |
| --- | --- |
| 硬件 | 独占同一组 4 张 B300；同一 host、CPU/NUMA 和互联 |
| Omni baseline | `2f289bb179d85671c116c10208b5c84b5c53f18f` |
| 模型 | `Lightricks/LTX-2.5-Diffusers`，revision `a6de4b5354f078db24d9cf4778c14846788aea3d` |
| Pipeline | `LTX2DistilledTwoStagePipeline`，T2V + 音频 |
| 形状 | 1920×1088，121 frames，24 fps，BF16，request concurrency 1 |
| 实际 schedule | Stage 1：8 个 Euler-ancestral steps；Stage 2：3 个 Euler refine steps；两段 sigma 列表显式传递 |
| Guidance / decoder | positive-only；Native DiffVAE；音频 48kHz stereo |
| Attention / codec | CUDNN_ATTN；H264 编码 preset ultrafast / threads 0 / CRF 18 |
| 负载 | 脚本内三个固定 prompt，测量 seeds 42/43/44 |
| Warmup | 每个新 server 两轮完整形状，使用 10042..10044 和 11042..11044，避免测量请求命中预热成品缓存 |
| Timing | `/v1/videos/sync` 从发送请求到完整 MP4 下载；包括 A/V 生成、编码与传输；不含模型加载、预热或 profiling |

这是一个固定 workload 的最佳 **tested** 配置，不是所有 LTX-2.5 variant、形状、吞吐场景的全局最优。Full / one-stage / I2V 需另立协议；不能切换 variant 或步数来完成本目标。

## 并行搜索空间

固定 CFG=1、Ring=1。Distilled 是 positive-only，增加 CFG ranks 不会产生真实 CFG 工作分片。

| TP | Ulysses | VAE tile degree | 搜索意义 |
| --- | --- | --- | --- |
| 1 | 4 | 1、2、4 | 主 SP 路径 |
| 2 | 2 | 1、2、4 | TP/SP 条件候选，必须实际完成三次完整质量验证 |
| 4 | 1 | 1、2、4 | TP 条件候选，必须实际完成三次完整质量验证 |

`TP × Ulysses = 4`；VAE degree 复用同一 worker WORLD，不再乘一次 GPU 数。每 cell 使用三个 fresh-server trials；OOM、退出、质量失败或来源漂移都不能进入 best 排名。单 worker 仅用作 canonical quality reference，不作为 30% 性能基线。

## 1. Remote 准备

在已经配好 CUDA / vLLM-Omni、独占四张 B300 的 Linux 环境中操作。若使用下面的无沙箱 Codex exec，必须是这次任务专用的隔离容器；不要在共享 host 上用 bypass 标志。

```bash
export LAB=/work/ltx25-b300-rsi
export OMNI_PY=/path/to/prepared-omni-venv/bin/python
export CUDA_VISIBLE_DEVICES=0,1,2,3   # 替换为实际已分配的 IDs/UUIDs
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$LAB"/artifacts "$LAB"/cache "$LAB"/guards
export HF_HOME="$LAB/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"

git clone --branch feat/cli-rsi-bootstrap https://github.com/david6666666/OmniRSI.git "$LAB/OmniRSI"
git clone https://github.com/vllm-project/vllm-omni.git "$LAB/baseline"
git -C "$LAB/baseline" fetch origin 2f289bb179d85671c116c10208b5c84b5c53f18f
git -C "$LAB/baseline" checkout --detach 2f289bb179d85671c116c10208b5c84b5c53f18f
git -C "$LAB/baseline" worktree add -b codex/ltx25-b300-30pct "$LAB/candidate" HEAD

"$OMNI_PY" -m pip install --no-deps "$LAB/OmniRSI"
command -v ffmpeg ffprobe codex
codex --version
nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
nvidia-smi topo -m
"$OMNI_PY" -m omnirsi doctor --repo "$LAB/baseline" --repo-type vllm_omni \
  --backend cuda --device-model B300 --device-ids "$CUDA_VISIBLE_DEVICES" --python "$OMNI_PY"
```

`OMNI_PY` 是已能运行目标 revision 的解释器，不是单独的 CPU control venv。该源版本的安装说明要求匹配 vLLM 0.31.0 source-wheel 路线、Diffusers 0.40.0、kernels 0.16.1 和相应 Transformers 范围；按固定源码的 CUDA 安装文档准备，不猜测 Torch 版本。保留 `pip freeze`、驱动、cuDNN/NCCL 和拓扑。

```bash
"$OMNI_PY" -m pip freeze > "$LAB/artifacts/pip-freeze.txt"
"$OMNI_PY" - "$LAB/artifacts/model-index.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
p = Path(hf_hub_download('Lightricks/LTX-2.5-Diffusers', 'model_index.json',
                        revision='a6de4b5354f078db24d9cf4778c14846788aea3d'))
content = p.read_bytes()
Path(sys.argv[1]).write_bytes(content)
print(json.dumps({'requested_revision': 'a6de4b5354f078db24d9cf4778c14846788aea3d',
                  'cached_path': str(p), 'sha256': hashlib.sha256(content).hexdigest(),
                  'class': json.loads(content).get('_class_name')}, indent=2))
PY
```

这个 HF pin 来自维护中的上游测试；本地没有独立下载验证。若 remote 无权限或缺资产，先解决加载与缓存，不能将其作为并行配置的性能结果。确认 baseline 和 candidate 的实际导入路径指向各自 checkout；helper 显式设置子进程 PYTHONPATH。

## 2. 冻结测量与质量工具

```bash
cp "$LAB/OmniRSI/examples/ltx25_b300.py" "$LAB/guards/"
cp "$LAB/OmniRSI/examples/ltx25_quality.py" "$LAB/guards/"
sha256sum "$LAB/guards/"*.py > "$LAB/artifacts/guards.sha256"
chmod a-w "$LAB/guards/"*.py
export VALIDATOR_SHA=$(sha256sum "$LAB/guards/ltx25_quality.py" | awk '{print $1}')

"$OMNI_PY" "$LAB/guards/ltx25_b300.py" trial --dry-run \
  --source-repo "$LAB/baseline" --cache-root "$LAB/cache" \
  --tp 1 --ulysses 4 --vae 4 --output "$LAB/artifacts/dry-run/result.json"
```

helper 使用固定模型 revision、两段 sigma、decoder 和 codec 参数。它还冻结 recipe/guidance/denoise/decoder/model-extra/serving/export 源文件哈希；优化不得修改这些保护路径。可以优化 transformer / ops / attention / 不改变语义的分布式算子。禁止改 guard、指标、阈值、参考输出、精度、步数、帧数、模型或成品缓存。

GPU cache 和产物在源码 checkout 外。HF assets/config 也要冻结，保留实际 cache snapshot/config 哈希，禁止 Codex 修改模型或安装依赖。每 trial 记录源码前后指纹、包版本、helper SHA 和全部 MP4 SHA；启动/失败日志保留。

## 3. Canonical 单 worker reference 与重复性

```bash
"$OMNI_PY" "$LAB/guards/ltx25_b300.py" trial \
  --source-repo "$LAB/baseline" --cache-root "$LAB/cache" \
  --single-reference --tp 1 --ulysses 1 --vae 1 \
  --output "$LAB/artifacts/reference/result.json"

"$OMNI_PY" "$LAB/guards/ltx25_b300.py" trial \
  --source-repo "$LAB/baseline" --cache-root "$LAB/cache" \
  --single-reference --tp 1 --ulysses 1 --vae 1 \
  --output "$LAB/artifacts/reference-repeat/result.json"

"$OMNI_PY" "$LAB/guards/ltx25_quality.py" \
  --reference-manifest "$LAB/artifacts/reference/result.json" \
  --candidate-manifest "$LAB/artifacts/reference-repeat/result.json" \
  --min-ssim 0.99 --min-psnr 35 --max-audio-nrmse 0.01 \
  --expected-validator-sha "$VALIDATOR_SHA" \
  --output "$LAB/artifacts/reference-repeatability.json"
chmod -R a-w "$LAB/artifacts/reference"
```

先确认 reference 自身完整且可重复。质量政策是本实验预先声明的保守门槛，不是官方标准：每帧 SSIM≥0.99、每帧 PSNR≥35dB；48kHz 两个音频声道的 NRMSE≤0.01；输出尺寸、实际帧数、FPS、音视频时长/起点和完整解码均通过。允许最多约两帧的 latent-grid/codec padding；极短音轨不能通过。它评估相对 reference 的一致性，不证明绝对感知质量；还需人工观看/试听和上游质量套件。

如果 reference 的重复性失败，停止并报告，而不是在看到候选收益后放宽阈值。

## 4. 搜索并冻结四 worker 最优配置

```bash
"$OMNI_PY" "$LAB/guards/ltx25_b300.py" search \
  --source-repo "$LAB/baseline" --cache-root "$LAB/cache" \
  --reference-manifest "$LAB/artifacts/reference/result.json" \
  --quality-script "$LAB/guards/ltx25_quality.py" \
  --include-tp-candidates --repetitions 3 \
  --output-dir "$LAB/artifacts/parallel-search"

mapfile -t BEST < <("$OMNI_PY" - "$LAB/artifacts/parallel-search/best.json" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))['best']
for key in ('tp', 'ulysses', 'vae'):
    print(b[key])
PY
)
export BEST_TP=${BEST[0]} BEST_U=${BEST[1]} BEST_VAE=${BEST[2]}
export BEST_PARALLEL="$BEST_TP,$BEST_U,$BEST_VAE"
cat "$LAB/artifacts/parallel-search/best.json"
sha256sum "$LAB/artifacts/parallel-search/best.json" > "$LAB/artifacts/best.sha256"
```

完整搜索是 9 cells × 3 trials；每 trial 有 6 个 full-shape warmups + 3 个计时请求，资源预算需预先考虑。不能同时改 baseline 或并行启动其他测量。TP 路径有实现不等于此 checkpoint 已合格；只有重复测试与 A/V quality 全通过才 eligible。若没有 eligible cell，不生成 best。

保留每个失败、所有原始请求延迟、quality 报告与 source/device/cache 数据。再次复测选中 winner，而不是拿搜索中的最幸运一次当最终基线。

## 5. 用 Codex 在固定 winner 上做性能优化

先导出已有方法，再在 **这次任务的独占 GPU 隔离容器** 中运行下面的 Codex 命令。该调用是 remote 的外部 Codex CLI；OmniRSI 的 `--agent codex` 不会自动启动 Codex。

若希望 Codex 负责整个实测过程，完成第1–2节环境与 guard 准备后即可运行本节命令；任务提示词会先执行尚未完成的 reference 与并行搜索，再读取 winner 进行优化。第3–4节也提供了可逐项手动执行的命令。

```bash
"$OMNI_PY" -m omnirsi context --repo-type vllm_omni --backend cuda \
  --scenario diffusion.video_generation --limit 5 \
  --output "$LAB/artifacts/diffusion-context.md"

{
  printf 'Campaign root: %s\nPrepared interpreter: %s\nAllocated devices: %s\n' "$LAB" "$OMNI_PY" "$CUDA_VISIBLE_DEVICES"
  cat "$LAB/OmniRSI/docs/prompts/ltx25-b300-optimize.md"
} | codex exec -C "$LAB/candidate" --sandbox danger-full-access --json \
  --output-last-message "$LAB/artifacts/codex-final.md" - \
  > "$LAB/artifacts/codex-events.jsonl"
```

共享 host 使用 interactive Codex 的 workspace-write 和必要批准；不要直接照搬无沙箱模式。当前模型选择沿用你的 Codex 配置。任务要求读取本文、best.json、固定 reference 和来源知识，先 profile，逐项提出可证伪假设，限定优化范围并保留每次失败证据。

## 6. 诊断与最终 30% AB/BA 验收命令

```bash
# Full-workload diagnostic only; diagnostic records have no performance metric.
"$OMNI_PY" "$LAB/guards/ltx25_b300.py" trial --diagnostic \
  --source-repo "$LAB/baseline" --cache-root "$LAB/cache" \
  --tp "$BEST_TP" --ulysses "$BEST_U" --vae "$BEST_VAE" \
  --output "$LAB/artifacts/best-profile/result.json"
```

该 DD pipeline 固定 8+3，诊断也不伪装成 2-step benchmark。profile 时间不能用于 30% 对比。

下面的命令可由 Codex 或操作者在候选代码稳定后执行；B/C 使用相同固定 winner、协议、依赖和参考。`--agent codex` 只是交接标签。

```bash
BASELINE_CMD="$OMNI_PY $LAB/guards/ltx25_b300.py trial --source-repo $LAB/baseline --cache-root $LAB/cache --tp $BEST_TP --ulysses $BEST_U --vae $BEST_VAE --output {result}"
CANDIDATE_CMD="$OMNI_PY $LAB/guards/ltx25_b300.py trial --source-repo $LAB/candidate --cache-root $LAB/cache --tp $BEST_TP --ulysses $BEST_U --vae $BEST_VAE --output {result}"
QUALITY_CMD="$OMNI_PY $LAB/guards/ltx25_quality.py --reference-manifest $LAB/artifacts/reference/result.json --omnirsi-run --expected-parallel $BEST_PARALLEL --expected-validator-sha $VALIDATOR_SHA --min-ssim 0.99 --min-psnr 35 --max-audio-nrmse 0.01 --output {result}"

"$OMNI_PY" -m omnirsi run \
  --repo "$LAB/baseline" --candidate-repo "$LAB/candidate" \
  --repo-type vllm_omni --backend cuda --device-model B300 \
  --device-ids "$CUDA_VISIBLE_DEVICES" --python "$OMNI_PY" \
  --scenario diffusion.video_generation --mode code --agent codex \
  --model Lightricks/LTX-2.5-Diffusers --model-revision a6de4b5354f078db24d9cf4778c14846788aea3d \
  --workload-id ltx25-dd2-1920x1088-f121-24fps-s8plus3-seeds42-44 \
  --baseline-command "$BASELINE_CMD" --candidate-command "$CANDIDATE_CMD" \
  --validation-command "$QUALITY_CMD" \
  --metric-key metrics.latency_ms --metric-unit ms --comparison-scope sync_video_audio_e2e \
  --direction minimize --repetitions 5 --min-improvement-pct 30 \
  --max-minutes 240 --output-dir "$LAB/artifacts/final-runs"
```

以上默认 `/work/...` 路径无空格；若使用其他路径，按 argparse 的 quoted argv 规则为 command 内部的路径加引号。每 trial 新 server、same full-shape warmup，按 AB/BA 交替重测。validation 从 `OMNIRSI_RESULT_PATH` 找到本轮 run，并检查 **所有 baseline/candidate trials**，不是只检查最后一个视频。

选中 winner 的新的五次 baseline 作为 `L0`；候选五次为 `L1`。只有完整请求与 source/guard/parallel/quality 全通过、且 `L1/L0≤0.70` 时，才接受“30%”。当前初版报告 descriptive median/range，不代表统计显著性。建议另用预先冻结的 held-out prompt/seed 复测泛化。

## 交付与失败处理

- 交付搜索全部 cells、winner、四卡拓扑、版本与源码、固定模型/资产、逐请求数据、profile、质量输出、HTML report、Codex JSONL、候选 diff 与 `git commit -s`。
- 禁止以减步、降分辨率/帧数、关闭音频、换 decoder/variant、量化、成品缓存或改变测量/质量规则完成目标。
- candidate 改 protected semantics 或依赖时，自动门禁失败；不要绕过它。需要新的独立协议/语义审阅，不能沿用本 30% 结论。
- Ctrl-C / timeout 只能清理此次拥有的 server/worker；helper 在 OmniRSI 中继承 trial group，直接运行则使用独立的 owned server group。禁止设备级 broad kill。
- 30% 未达到就报告未达到；没有有效 best 就停止；缺证据保留 INCONCLUSIVE，不编造已验证性能。

## 固定来源

- [LTX-2.5 recipe](https://github.com/vllm-project/vllm-omni/blob/2f289bb179d85671c116c10208b5c84b5c53f18f/recipes/LTX/LTX-2.5.md)
- [recipes/schedules](https://github.com/vllm-project/vllm-omni/blob/2f289bb179d85671c116c10208b5c84b5c53f18f/vllm_omni/diffusion/models/ltx2/ltx2_recipes.py)
- [TP implementation](https://github.com/vllm-project/vllm-omni/blob/2f289bb179d85671c116c10208b5c84b5c53f18f/vllm_omni/diffusion/models/ltx2/ltx2_transformer.py)
- [CUDA source installation](https://github.com/vllm-project/vllm-omni/blob/2f289bb179d85671c116c10208b5c84b5c53f18f/docs/getting_started/installation/gpu/cuda.inc.md)
- [Codex non-interactive documentation](https://learn.chatgpt.com/docs/non-interactive-mode)；[CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
