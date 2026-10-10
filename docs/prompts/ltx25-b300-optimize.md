# B300×4 LTX-2.5：固定最佳并行后的20%性能任务

本任务在操作者已准备的 remote GPU 隔离环境中执行。消息开头提供 Campaign root、Prepared interpreter 与已分配 GPU IDs。读取 Campaign root/OmniRSI/docs/ltx25-b300-test-plan.md，沿用里面的实际 CLI 命令。不要猜测尚未实现的 OmniRSI flags。

目标：在已通过完整质量验证的四 worker winner 上，以新的5组 AB/BA 为基线，达到端到端延迟下降至少20%。未达到就明确报告未达到；这不是需要伪造的必达数字。

1. 核对 exactly4 B300、独占资源、解释器和导入路径、源码pin2f289bb179d85671c116c10208b5c84b5c53f18f、模型pin、FFmpeg和依赖。只操作候选checkout及本任务artifact/cache，不修改baseline。
2. 读取并核对 guards.sha256、reference/result.json、reference-repeatability、parallel-search/search.json 与 best.json。reference 缺失时，先按 Test plan 第3节生成单 worker reference 并验证独立重复性；search 尚未执行时，按第4节运行全部9个条件候选的3次测量并冻结 winner。源码漂移、质量失败或无 eligible winner 时停止。别在测量同时改源码。
3. 固定 best 的TP/Ulysses/VAE tuple；固定DDtwoStage、BF16、1280×704（720p档位，64对齐）、121frames、24fps、8+3sigmas、positive-only、NativeDiffVAE、48k stereo、CUDNN_ATTN、编码参数、三条prompt/seeds和独立warmup seeds。
4. 阅读 diffusion-context.md，区分 imported 上游经验与本地实测。先做同形状2+1steps短schedule profiler诊断（可调），稳定候选按需补完整8+3profile，定位DiT/FFN/norm/RoPE/通信/布局成本。诊断结果不能作performance基线。
5. 每次只实现一个可证伪假设。允许语义保持的 transformer/ops/attention/分布式算子优化；保护路径由helper列出，禁止修改recipe、scheduler、guidance、request、decoder、server/export、模型文件或已安装依赖。
6. 禁止改helper、validator、阈值、best/reference/冻结协议；禁止memoize/重放最终视频或音频、伪造timing、减步、关音频、改形状/精度或切换模型variant。缓存静态计算时记录适用范围、request生命周期与新seed正确性。
7. 先跑相关CPU/CUDA正确性测试和一次完整A/V trial，质量通过后再做完整5组OmniRSI验证。所有trial都检查音视频完整性、每帧SSIM/PSNR、双声道NRMSE、来源、helper/validator和固定parallel。不得只挑有利的prompt或最后一个trial。
8. 预算：最多8个代码假设、最多8小时实测时间。超时/预算耗尽时完成已启动trial的受控清理并保留全部证据。不能通过松门槛或换协议继续凑20%。不要开并行GPU测量。
9. 只有L1/L0<=0.80且全部门禁通过，才写成功。记录median/range、所有原始sample、逐prompt变化和质量；不称统计显著性或所有LTX场景的全局最优。额外held-out prompt/seed若没有验证需列为未验证。
10. 输出artifact报告和候选diff，所有提交用git commit -s。默认不push、merge、修改账号/权限或发布生产；操作者明确授权更新已有PR时可push该PR的head分支并添加带证据的comment，仍不merge或发布生产。最终回复列出路径、best、收益、质量、失败/限制、未验证项。保存Codex JSONL与最终报告。
