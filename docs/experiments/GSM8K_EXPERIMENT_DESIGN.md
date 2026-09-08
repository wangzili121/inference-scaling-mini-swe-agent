# 运行与评测

本页说明 AR-LLM 与 dLLM 的训练、推理和评测入口。算法目标、概率修正、奖励定义和执行实现集中在
[算法文档](../methods/ALGORITHMS.md)；安装与解释器配置见[README](../../README.md#安装)。
已完成的实验分别汇总在[算法设计与准确率](../reports/GSM8K_3090_ALIGNED_RESULTS.md)和
[推理成本与执行效率](../reports/RTX3090_ROLLOUT_INFRA.md)中，各报告独立列出所用设置。

## 数据与配置

配置位于 [`configs/`](../../configs/)。`gsm8k_quick.toml` 用于短预算检查，`gsm8k_3090_aligned.toml`
用于 Qwen2.5-1.5B，`gsm8k_llada_moe_3090.toml` 提供 LLaDA-MoE 的模型与算法参数。GRPO 训练单独读取
`gsm8k_grpo.toml`。模型路径、revision、数据子集、随机种子与生成长度均由配置和 CLI 参数决定。

准备脚本下载固定版本的公开 GSM8K 数据与模型。AR 使用 Qwen2.5-1.5B 作为基础模型，0.5B 模型可用于
候选或 rollout proposal；dLLM 使用 LLaDA-MoE，低层 proposal 由同一模型的分层执行提供。

数值参考值 verifier 会读取数据集标准答案，适合共享奖励目标诊断。自一致性、序列 logprob 和 Consilience
使用模型自身的输出或概率。自定义 verifier 通过 `--verifier-config` 选择；
`requires_reference = false` 时，入口向 verifier 提供提示与生成，不传入标准答案。

<a id="method-labels"></a>
## 方法标识

| 方法 | AR-LLM 标识 | dLLM 标识 |
| --- | --- | --- |
| 基础模型采样 | `base` | `base` |
| Beam search | `beam` | `block_beam` |
| 多次生成后选择 | `best_of_n` | `best_of_n` |
| 幂目标 MH | `mh` | `trajectory_power_mh` |
| 条件 IS | `conditional_is` | `conditional_is` |
| 低成本 rollout proposal | `conditional_is_small_proposal` | `conditional_is_reduced_layer_proposal` |
| 未校正 rollout 加权 | `conditional_is_small_proposal_uncorrected` | `conditional_is_reduced_layer_proposal_uncorrected` |
| RL 参数随机采样 | `rl_sample` | `vrpo_sample` |
| RL 参数贪心解码 | `rl_greedy` | `vrpo_greedy` |
| verifier 奖励 MH | `verifier_mh` | `verifier_mh` |
| verifier 奖励 IS | `verifier_conditional_is` | `verifier_conditional_is` |

RL 参数来自 GRPO 或 VRPO 训练；随机采样使用配置的采样策略，贪心解码每步取最大概率项。
完整标识与配对关系由 [`methods.py`](../../experiments/shared/methods.py) 维护，各入口的 `--help` 列出
可选值。迭代 IS、动态候选等研究方法需要显式选择，采用条件见[非默认方案记录](../methods/ALGORITHMS.md#alg-nondefault-notes)。

## 统一入口

以下命令从仓库根目录执行。默认运行 AR-LLM；解释器依次取 CLI 参数、`AR_PYTHON` / `DLLM_PYTHON`
环境变量和当前 Python。

低成本功能检查包含数据准备、一次 GRPO 更新和短预算推理：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage all --profile smoke --tag local-qwen
```

完整的 AR 训练与推理：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage all --profile full --tag qwen-full
```

使用已有模型，只运行指定方法与组件：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage inference --profile full --tag qwen-is-mh `
  --ar-methods base mh conditional_is `
  --components quality replay async --limit 32
```

LLaDA-MoE 在满足显存需求的机器上使用相同入口：

```powershell
python experiments\run_reproduction.py `
  --family dllm --stage all --profile full --tag llada-full `
  --dllm-python $env:DLLM_PYTHON
```

`--family both` 依次调度两侧。`--stage all` 将本次训练产生的适配器显式传给后续推理任务；只运行推理时，
RL 方法要求配置中的适配器已经存在。`--dry-run` 写出命令清单，供启动前检查路径、解释器与预算。

| 参数 | 作用 |
| --- | --- |
| `--limit` | 推理题目数量 |
| `--train-limit`、`--max-train-steps` | 训练样本与更新预算 |
| `--max-completion-length` | 训练补全长度 |
| `--passk-limit`、`--passk-draws` | pass@k 题目与独立重复数量 |
| `--ar-methods`、`--dllm-methods` | 两侧的具体推理方法 |
| `--ar-mh-suffix-schedule` | `uniform`、`inverse_length` 或 `multiscale`；统一入口默认 `multiscale` |
| `--verifier-config` | 独立奖励配置 |
| `--output-root` | 调度清单和组件汇总的输出目录 |

使用不同模型配置或后端时，可直接调用模型族入口：

```powershell
python -m experiments.arllm.run_arllm_suite `
  --config configs\gsm8k_3090_aligned.toml `
  --stage inference --profile full --methods base mh conditional_is `
  --components quality --backend transformers --tag custom-ar
```

<a id="infra-labels"></a>
<a id="replay-labels"></a>
## 组件与比较对象

| 组件 | 运行内容 | 比较对象 |
| --- | --- | --- |
| `quality` | 所选方法的单次生成评测 | 相同题目和生成长度的 Base |
| `matched_target` | 固定 verifier 下的 MH、IS 与 RL | 同一奖励定义与尺度 |
| `replay` | 纯新生成、已有历史与候选复用 | 相同候选与总 rollout 预算 |
| `async` | 连续批处理 | 相同请求和随机种子的逐提示执行 |
| `passk` | 每题独立重复生成 | 相同独立重复数 |
| `distribution` | 经验答案分布与累计计算量 | 共享奖励目标 |
| `dynamic_is` | 动态 proposal 与预算分配 | 固定候选 proposal 与固定分配 |
| `infra` | rollout 调度、流式奖励、MH 复用与 SMC | 同一算法的基础执行路径 |
| `ablations`、`budget_curve`、`length_ablation` | 参数、预算与长度扫描 | 每次固定其余参数 |
| `vllm` | AR 后端比较 | 相同模型、数据与数值类型的 Transformers |

`full` 默认包含前六项；其余组件通过 `--components` 显式选择。replay 最终估计记录预留后只使用一次；
候选重复出现时，共享该匹配键的库存约束。历史库构建、初始估计和在线推理成本分别记录。

## 统计与成本

质量统计包括单次生成准确率、Wilson 区间、题目级配对自助法区间与独立重复的 pass@k。经验答案分布使用
总变差距离（TV）和 Jensen–Shannon 散度。IS 记录权重 ESS 与复用数；MH 记录更新数、接受率和改变的 token 数。
统计实现位于 [`shared/statistics.py`](../../experiments/shared/statistics.py) 与
[`shared/metrics.py`](../../src/inference_scaling/shared/metrics.py)。

模型 $`j`$ 的前向 FLOPs 按参数量 $`N_j`$ 与实际参与前向计算的 token 位置数 $`S_j`$ 估算：

```math
\widehat F_j=2N_jS_j.
```

计数覆盖前缀预填充、逐 token 生成、序列重评分与草稿验证。基础模型和 proposal 模型分别记录后求和；该估算
省略注意力中随长度平方增长的项及逐元素算子。GRPO/VRPO 另外记录当前策略前向、反向、参考评分与适配器更新。

墙钟排除模型和数据加载，包含调度、评分及本次任务的收尾。比较时同时报告“优化路径成本 / 基线路径成本”
和采用的基线；小于 1 表示成本降低。后台预生成、缓存构建与首次查询的成本单列，避免把预先计算视作零成本。

## 输出与续跑

| 内容 | 默认位置 |
| --- | --- |
| 根级调度清单 | `results/reproduction/<tag>/reproduction_manifest.json` |
| AR 逐题记录 | `results/gsm8k/<profile>/` |
| AR 汇总与子清单 | 根级入口的 `<output-root>/arllm/`；独立入口的 `--summary-root` |
| dLLM 组件结果 | 根级入口的 `<output-root>/dllm/` |
| 模型与适配器 | `models/` 或训练参数指定的位置 |

输出目录由入口按需创建，`results/` 整体由 Git 忽略。仓库保留可重新运行的代码与配置，历史实验产物不作为
测试数据或运行前置条件。单元测试在临时目录构造所需样本。

逐题 JSONL 按配置标识续写，pass@k 使用独立任务分块。清单记录有效配置、数据行号、模型 revision 和实现哈希；
相同标签与配置可续跑。算法、模型或预算改变时使用新的 `--tag`。底层脚本和汇总器的参数可从统一入口的
`--dry-run` 输出取得。
