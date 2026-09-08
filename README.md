# inference_scaling

本仓库同时实现自回归语言模型（AR-LLM）和掩码扩散语言模型（dLLM）的训练与推理扩展。AR-LLM
使用 Qwen2.5 与组相对策略优化（Group Relative Policy Optimization，GRPO）；dLLM 使用 LLaDA-MoE 与
方差缩减偏好优化（Variance-Reduced Preference Optimization，VRPO）。两侧共享 GSM8K 数据、奖励、统计量、计算量记录和
可续跑调度，并分别实现 Metropolis--Hastings（MH）、重要性采样（IS）与 rollout replay。

## 目标分布与方法

给定提示 $`x`$、基础模型分布 $`p(y\mid x)`$、序列奖励 $`r(y)`$ 和奖励温度 $`\tau`$，考虑在完整
序列分布上求解 KL 正则化目标：

```math
\max_{\pi(\cdot\mid x)}
\left\{
\sum_y \pi(y\mid x)r(y)
-\tau D_{\mathrm{KL}}\!\left(\pi(\cdot\mid x)\,\|\,p(\cdot\mid x)\right)
\right\},
\qquad \sum_y\pi(y\mid x)=1.
```

第一项提高期望序列奖励，第二项限制新分布偏离基础模型；$`\tau`$ 是两者的权衡系数。对归一化约束加入
拉格朗日乘子后，一阶条件为

```math
r(y)-\tau\left(\log\frac{\pi(y\mid x)}{p(y\mid x)}+1\right)+\lambda=0.
```

因此 $`\pi(y\mid x)\propto p(y\mid x)\exp\{r(y)/\tau\}`$，归一化后得到仓库采用的主要目标分布：

```math
\pi_r(y\mid x)
=\frac{p(y\mid x)\exp\{r(y)/\tau\}}
       {\sum_{y'}p(y'\mid x)\exp\{r(y')/\tau\}}.
```

该闭式解按照奖励重新分配基础模型已有完整序列的概率质量。仓库直接对这一
分布进行采样或近似，提供以下路径：

| 路径 | 核心操作 | off-policy / replay 处理 | 主要实现 |
| --- | --- | --- | --- |
| [后缀 MH](docs/methods/ALGORITHMS.md#alg-power-mh) | 重生成随机后缀或扩散块，再按 Hastings 比接受或拒绝 | 提议分布（proposal）的正反概率进入接受率 | [共享接受核](src/inference_scaling/shared/mh.py)、[AR 适配](src/inference_scaling/arllm/algorithms/mh.py)、[dLLM 适配](src/inference_scaling/dllm/algorithms/search.py) |
| [条件 IS](docs/methods/ALGORITHMS.md#alg-conditional-is) | 为下一个生成块产生候选，用 rollout 估计条件奖励权重后重采样 | 补全来自其他模型时乘 $`p/q`$ | [AR 实现](src/inference_scaling/arllm/algorithms/conditional_is.py)、[dLLM 实现](src/inference_scaling/dllm/algorithms/is_sampling.py) |
| [rollout replay](docs/methods/ALGORITHMS.md#alg-base-replay) | 复用历史补全，并保留本次新生成的 rollout 以覆盖支持集 | 使用实际生成分布的概率和新样本校正项 | [AR replay](src/inference_scaling/arllm/algorithms/base_replay.py)、[dLLM replay](src/inference_scaling/dllm/replay.py) |
| [动态候选](docs/methods/ALGORITHMS.md#alg-dynamic-is) | 由辅助提议分布生成候选，并按方差与成本分配 rollout | 外层 $`p/q_c`$ 修正候选来源 | [显式研究实现](src/inference_scaling/experimental/arllm/dynamic_is.py) |
| [可枚举候选 logit adjustment](docs/methods/ALGORITHMS.md#alg-logit-adjustment) | 将估计条件权重的对数加到基础候选 logits，再在完整候选集上归一化 | 可直接使用新生成、off-policy 或 replay 条件权重 | 理论参考；当前没有 CLI、代码实现或实验结果 |

共享算法层不依赖模型的生成方向。条件 IS 使用统一的逐步候选、rollout 权重与重采样接口；MH 使用统一的
未归一化目标概率的对数差、正反 proposal 比和接受/拒绝核。AR-LLM 与 dLLM 目录只实现 token 后缀、掩码块或扩散
轨迹的生成和概率评分。实验方法及其适用组件集中登记在
[`experiments/shared/methods.py`](experiments/shared/methods.py)，两侧入口、配对协议和汇总程序不再分别维护
方法名称清单。

训练对照采用 GRPO 与 [VRPO](https://arxiv.org/abs/2505.19223)。VRPO 以掩码扩散的证据下界（ELBO）
代替序列对数似然：每个偏好对采样 8 个独立掩码比例，每个比例采样 1 个掩码，并让当前策略与冻结的参考模型
使用相同掩码。LoRA 适配器与关闭适配器后得到的参考模型共同使用同一份已加载基础模型。

当前 Qwen 默认 MH/IS 的完整步骤、模型职责和参数表，以及初始估计与最终估计分离的 IS、流式奖励、SMC
多树搜索、两阶段延迟接受 MH、历史后缀 proposal、批处理、KV 复用和 vLLM 后端，均集中在同一份
[算法基础、原理与实现文档](docs/methods/ALGORITHMS.md)中按“目标—算法—实现—误差与成本”组织。

## 奖励与 verifier 配置

MH、IS 与 replay 的算法层统一接收
`reward(prompt_tokens, completion_tokens) -> float`，或保持相同逐序列定义的批量版本。数据集读取、文本解析、
远程服务和模型族均不进入算法实现。外部 verifier 由顶层 `[verifier]` 表选择；默认 GSM8K 配置使用数值
参考值插件，但该插件只是一个可替换实现：

```toml
[verifier]
provider = "python"
name = "numeric_reference"
factory = "inference_scaling.shared.evaluation.numeric:build_numeric_reference_verifier"
requires_reference = true

[verifier.options]
correct_reward = 1.0
incorrect_reward = 0.0
unparseable_reward = 0.0
```

`python` provider 从可信的本地模块加载工厂函数。工厂接收 `context` 和 `[verifier.options]`，返回
`score(prompt, completion)` 对象或等价可调用对象；可选的 `score_batch(inputs)` 用于批量服务。
`requires_reference = false` 时，实验入口不会把数据集参考值传给 verifier。配置、工厂路径和参数共同生成稳定
版本号，replay 只复用版本一致的奖励记录。核心接口位于
[`shared/verifier.py`](src/inference_scaling/shared/verifier.py)，独立配置示例位于
[`configs/verifiers/`](configs/verifiers/)；所有统一入口都接受 `--verifier-config`：

```powershell
python -m experiments.arllm.gsm8k_reproduction `
  --method verifier_conditional_is `
  --verifier-config configs\verifiers\gsm8k_numeric_reference.toml `
  --limit 1 --tag verifier-check
```

GRPO 的批量奖励适配器也读取同一 `[verifier]` 表；`gold_answer` 仅在
`requires_reference = true` 时交给 verifier。VRPO 偏好数据按 verifier 分数选择最高与最低的生成；默认配置把
公开训练集解答作为一个额外候选并同样评分，设置
`vrpo_training.include_reference_completion = false` 可完全排除该候选。训练与推理可使用同一
`--verifier-config`，也可在各自配置文件中选择不同 verifier。

AR-LLM 还实现与外部 verifier 分离的完整序列对数概率奖励：

```math
r_{\log p}(x,y)=c\log p(y\mid x),
\qquad
p(y\mid x)\exp\{r_{\log p}(x,y)/\tau\}
=p(y\mid x)^{1+c/\tau}.
```

因此目标为 $`p^\alpha`$ 时可取 $`c=(\alpha-1)\tau`$。`Best-of-N` 直接复用生成时保存的 token
对数概率；条件 IS 与迭代 IS 通过后端的批量序列评分计算该奖励。这里的 $`p`$ 是配置实际采用的完整支持
采样策略。直接设置 $`c=1`$ 时指数是 $`1+1/\tau`$，并不固定为 2；例如 $`\tau=0.1`$ 时指数为 11。
MH 对同一目标直接使用 `mh --mh-alpha <alpha>`，无需把 logprob 再作为奖励评分一次。该奖励模式要求模型后端
返回精确 token 对数概率：

```powershell
python -m experiments.arllm.gsm8k_reproduction `
  --method conditional_is --conditional-reward sequence_log_probability `
  --reward-temperature 0.5 --logprob-reward-scale 0.5 `
  --limit 1 --tag power-two-is
```

AR-LLM 还支持 [Consilience](https://arxiv.org/abs/2608.09898) 的置信度轨迹奖励。它只读取同一模型逐 token
的 top-$`K`$ 对数概率，不读取参考答案或外部 verifier。默认设置跳过序列开头 5%，分别取随后 20% 与末尾
20% 的置信度均值，并计算“末段均值减去 3 倍首段均值”。该原始分数不做候选组内归一化，因此是固定的
逐序列奖励，可直接进入条件 IS、迭代 IS 和 rollout replay：

```powershell
python -m experiments.arllm.gsm8k_reproduction `
  --method conditional_is --conditional-reward consilience `
  --consilience-top-k 5 --consilience-window-fraction 0.2 `
  --consilience-skip-fraction 0.05 --consilience-initial-penalty 3 `
  --limit 1 --tag consilience-is
```

Qwen2.5-1.5B-Instruct 默认对完整生成计算该分数。具有显式推理结束标记的模型可用
`--consilience-reasoning-end-text` 排除标记及其后的最终结论；vLLM 路径需要配置 Transformers 精确评分后端。
完整公式、实现边界与成本见[已实现的奖励信号](docs/methods/ALGORITHMS.md#alg-rewards)。

## 文档

| 文档 | 内容 |
| --- | --- |
| [算法基础、原理与实现](docs/methods/ALGORITHMS.md) | 默认 Qwen MH/IS 完整流程、数学目标、模型职责、参数、关键代码、直观收敛说明、执行优化和 vLLM 配置 |
| [运行与评测](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md) | 数据配置、方法标识、训练与推理命令、统计量和输出目录 |
| [算法设计与准确率](docs/reports/GSM8K_3090_ALIGNED_RESULTS.md) | 固定实验设置下的准确率、pass@k、奖励与 proposal 对照，以及结果适用范围 |
| [推理成本与执行效率](docs/reports/RTX3090_ROLLOUT_INFRA.md) | 批处理、IS/MH 复用和奖励调度的墙钟、分模型 FLOPs、建库与设计成本 |
| [非默认方案记录](docs/methods/ALGORITHMS.md#alg-nondefault-notes) | 已筛选方案的主要成本问题与适用条件 |

## 实现范围

| 模型族 | 模型与训练对照 | 推理组件 | 执行接口 |
| --- | --- | --- | --- |
| AR-LLM | Qwen2.5-1.5B 主模型与 GRPO；0.5B 可作 proposal/rollout | MH、条件 IS、replay、可选研究方法 | Transformers 与 vLLM；两种模型的计算量分别记录 |
| dLLM | LLaDA-MoE-7B-A1B 与 VRPO | 分块生成、轨迹 MH、条件 IS 与 replay | 批量 Transformers；提供轻量测试和大显存机器入口 |
| 公共层 | 与模型无关 | 逐步候选、IS/replay 权重、MH 接受核、预算分配、SMC、统计与计算量记录 | AR/dLLM 共用同一实现 |

统一入口默认使用 `multiscale` 后缀 MH。replay 要求历史记录与当前提示、模型和采样策略匹配；IS 的最终估计
记录还必须尚未使用。候选缓存与连续批处理可复用已有请求，历史库构建成本单独统计。具体执行顺序见
[默认 MH 与 IS](docs/methods/ALGORITHMS.md#alg-qwen-default-mh)。

版本控制保留代码、配置、测试、使用文档和两份精选实验报告。运行产生的原始数据、汇总、日志和清单写入
`results/`，由 Git 统一忽略。报告分别讨论算法质量与执行成本，非默认方案的简要结论集中在算法文档中。

## 安装

AR-LLM 与官方 LLaDA-MoE 使用不同的 Transformers 版本。单独运行一侧时可直接使用当前 Python；完整成对
运行时使用两个解释器。

### 当前 Python

AR-LLM 依赖：

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,gpu,training]"
```

LLaDA-MoE 与 VRPO 依赖应安装到另一个 Python，或在只运行 dLLM 时安装到当前 Python：

```powershell
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev,dllm,dllm-training]"
python -m experiments.dllm.download_llada `
  --config configs\gsm8k_llada_moe_3090.toml --source modelscope
```

### 已有的 `.venv`

仓库根目录已有 `.venv` 时可直接作为控制器、dLLM 解释器或测试解释器，无需激活：

```powershell
.\.venv\Scripts\python -m pip install -e ".[dev,dllm,dllm-training]"
.\.venv\Scripts\python -m experiments.dllm.run_llada_suite --profile smoke
.\.venv\Scripts\python -m pytest
```

### 两个显式解释器

解释器可以来自系统安装、已有 `.venv`、Conda 或其他 Python 安装。变量值既可为绝对路径，也可为 `PATH`
中的可执行文件名：

```powershell
$env:AR_PYTHON = "C:\path\to\ar-python.exe"
$env:DLLM_PYTHON = ".\.venv\Scripts\python.exe"

& $env:AR_PYTHON -m pip install -e ".[dev,gpu,training]"
& $env:DLLM_PYTHON -m pip install -e ".[dev,dllm,dllm-training]"
```

### Linux / WSL2 vLLM

vLLM `0.25.x`--`0.26.x` 使用 Linux GPU wheel，并按官方 wheel 要求安装
PyTorch `2.11.0`。建议使用独立环境，避免改变已有训练环境中的 PyTorch。Windows 主机在 WSL2 的
Linux 文件系统中使用兼容的 Python：

```bash
python3.12 -m pip install --upgrade pip
python3.12 -m pip install -e ".[dev,vllm]"
```

幂目标 MH 在 vLLM `0.26.x` 上可把 proposal 概率和基础模型概率合并到同一次解码，省去生成后的整段
重评分。该路径使用同步入口，且不与 speculative decoding 同时启用：

```bash
python -m experiments.arllm.gsm8k_reproduction \
  --config configs/gsm8k_3090_aligned.toml \
  --backend vllm-sync --method mh --vllm-mh-fused-logprobs \
  --tag mh-fused --limit 32
```

实现约束、概率记账和统计字段见[算法与实现文档](docs/methods/ALGORITHMS.md#infra-vllm)。

## 统一复现入口

[`run_reproduction.py`](experiments/run_reproduction.py) 调度两侧的准备、训练和推理，默认只运行 AR-LLM。
AR 默认配置使用 Qwen2.5-1.5B；dLLM 通过 `--family dllm` 或 `--family both` 显式选择。两个 Python 路径分别
指向上述解释器。AR 的低成本功能检查（`smoke`）使用 1 题、缩短预算和一次 GRPO 更新。显式选择 dLLM 时，`smoke` 执行
CPU VRPO 反向传播、临时 LoRA 保存与重新加载检查；真实 LLaDA 推理子进程结束后释放模型显存。

解释器选择顺序为：CLI 的 `--ar-python` / `--dllm-python`、环境变量 `AR_PYTHON` / `DLLM_PYTHON`、
启动统一入口的当前 Python。单侧运行可省略两个解释器参数：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage all --profile smoke --tag local-qwen
```

环境变量方式无需在命令中重复路径：

```powershell
python experiments\run_reproduction.py `
  --family both --stage all --profile smoke --tag local-check `
  --ar-methods base mh conditional_is rl_sample `
  --dllm-methods base trajectory_power_mh conditional_is_reduced_layer_proposal `
  --components quality replay
```

大显存机器上的完整训练和推理使用相同入口。dLLM 阶段依次构造公开训练集偏好对、续跑 VRPO LoRA、加载
适配器，并运行配置中的推理方法；`--stage all` 会先下载或校验固定版本的 LLaDA 权重。AR 阶段依次
准备数据与权重、续跑 GRPO 和运行所选实验族，并把本次训练输出的适配器路径显式传给质量、pass@k、
消融和分布诊断，避免误用配置文件中的旧适配器。推理阶段显式选择 `vrpo_sample` 或 `vrpo_greedy` 时会
加载已有适配器；适配器不存在时入口在启动模型前报错。

Qwen2.5-1.5B 正式路线使用：

```powershell
python experiments\run_reproduction.py `
  --family arllm --stage all --profile full --tag qwen15b-full `
  --ar-python $env:AR_PYTHON
```

主要 CLI 参数：

| 参数 | 作用 |
| --- | --- |
| `--family arllm\|dllm\|both` | 运行一侧或成对运行 |
| `--stage prepare\|train\|inference\|all` | 选择模型准备、RL 训练、推理或完整流程 |
| `--profile smoke\|full` | 低成本实现检查或正式配置 |
| `--ar-methods ...`、`--dllm-methods ...` | 选择具体推理方法 |
| `--ar-mh-suffix-schedule ...` | 选择 AR-MH 后缀分布；默认值为 `multiscale`，`uniform` 用于基线复现 |
| `--components ...` | 选择质量、matched target、replay、动态 IS、异步、pass@k、消融、infra 等实验族 |
| `--verifier-config ...` | 用独立 TOML 文件替换外部 verifier，不修改数据集或算法配置 |
| `--ar-python ...`、`--dllm-python ...` | 覆盖环境变量与当前解释器 |
| `--limit`、`--max-train-steps` 等 | 覆盖样本数和训练预算 |
| `--dry-run` | 只写入清单并打印子命令，不启动训练或推理 |

`full` 默认调度 `quality`、`matched_target`、`replay`、`async`、`passk` 和
`distribution`。`dynamic_is`、`ablations`、`budget_curve`、`length_ablation`、`infra` 与 `vllm` 只在
`--components` 中显式指定时运行；它们用于研究消融或特定后端验证。dLLM 使用分块 beam、反向轨迹 MH、
低层 proposal、轨迹 replay、分块 SMC 与 VRPO 对应 AR 的 token 级方法。
AR 统一入口将 `multiscale` 传给质量与 pass@$`k`$ 的 MH 路径。replay 入口将建库时已经生成的基础模型候选
直接交给在线选择，避免第二次生成同一候选；连续批处理仍由 `async` 组件和执行后端承担。
方法标识、配对关系与各组件统计量见[运行与评测](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md#method-labels)。

两侧也可独立启动：

```powershell
& $env:AR_PYTHON -m experiments.arllm.run_arllm_suite `
  --stage all --profile full --tag full-ar

& $env:DLLM_PYTHON -m experiments.dllm.run_llada_suite `
  --profile full --vrpo train --tag full-dllm
```

所有入口写入命令清单和已完成子任务数。模型族入口位于 `experiments/arllm/` 与 `experiments/dllm/`，仓库根级
实验目录只保留成对调度入口。完整统计定义见
[运行与评测](docs/experiments/GSM8K_EXPERIMENT_DESIGN.md)。

## 测试与目录

```powershell
python -m pytest

# 或使用仓库中已有的解释器
.\.venv\Scripts\python -m pytest
```

| 路径 | 内容 |
| --- | --- |
| `src/inference_scaling/arllm/` | AR-LLM 的 MH、IS、replay、Transformers 与 vLLM 后端 |
| `src/inference_scaling/dllm/` | LLaDA-MoE 的分块生成、MH、IS、replay 与 VRPO |
| `src/inference_scaling/shared/` | 两侧共用的逐步生成、IS/replay 权重、MH 接受核、数据评测、随机数和计算量记录 |
| `src/inference_scaling/experimental/` | 保留但不由默认入口导入或调度的研究实现 |
| `configs/` | 模型、数据与预算配置 |
| `experiments/shared/` | 两侧共用的组件清单、统计量、配置标识、可续跑调度和结果文件管理 |
| `experiments/arllm/`、`experiments/dllm/` | 两侧独立复现入口与模型特定训练脚本 |
| `experiments/run_reproduction.py` | 成对调度 AR-LLM 与 dLLM 的统一入口 |
| `tests/` | 分布、实现一致性和结果处理测试 |
| `docs/` | 算法原理与实现、运行说明，以及算法质量和执行成本两份报告 |
| `results/` | 运行时生成的原始数据、汇总和清单，Git 忽略 |
| `online-speculation/` | 独立的在线推测解码项目，使用其目录内的说明与入口 |

公共算法接口位于 `inference_scaling.shared`；模型特定代码只负责生成状态、proposal 与概率评分。
