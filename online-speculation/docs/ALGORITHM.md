# 共享骨干的并行起草与在线续训

本文介绍一条完整的生成管线：自回归语言模型提供目标概率与历史缓存，双向注意力分支并行预测一块候选，概率校正提交连续前缀；推理期间积累的验证反馈用于更新起草注意力后段。正文中的公式对应本仓实现，实验条件与数值集中于 [RESULTS.md](RESULTS.md)。

## 1. 自回归概率与投机解码

### 1.1 下一枚 token 的分布

分词器把文本表示为词表中的整数序列 $x_1,\ldots,x_T$。词表大小记为 $V$；一个 token 可以对应一个字、一个词或词的一部分。自回归模型在位置 $t$ 读取前缀 $x_{\le t}$，输出一个长度为 $V$ 的分数向量 $u_t$，称为 logits。softmax 把分数变为概率：

$$
p_\theta(v\mid x_{\le t})
=\frac{\exp u_t(v)}{\sum_{w=1}^{V}\exp u_t(w)}.
$$

$\theta$ 表示模型参数，$v$ 表示下一枚 token 的取值。整段序列的概率由条件概率相乘得到：

$$
p_\theta(x_{1:T})
=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

这里把序列起始标记视为已给定的上下文。

下一项预测训练（NTP）最小化

$$
\mathcal L_{\mathrm{NTP}}
=-\sum_{t=1}^{T-1}\log p_\theta(x_{t+1}\mid x_{\le t}).
$$

训练序列已经给定，因果注意力使第 $t$ 行只读取 $x_{\le t}$。因此一次前向可以同时计算所有位置的训练项。生成时，$x_{t+1}$ 需要从当前分布中抽取，随后进入 $x_{t+2}$ 的条件上下文，形成逐 token 的串行依赖。

温度、top-k、top-p 共同定义实际的采样规则。本文用 $\mathcal S$ 表示这套概率变换，用 $p=\mathcal S(u^{\mathrm{AR}})$ 表示最终目标分布。温度为零时采用最大分数对应的 token，即贪心生成。所有候选验证都以同一套 $\mathcal S$ 为准。

### 1.2 候选概率与并行验证

给定已提交的上下文 $s$，起草器提出 $m$ 个候选 $y_1,\ldots,y_m$。其分布记为

$$
q_i(v)=q(v\mid s,y_{<i}).
$$

这是统一记法。独立并行起草器的 $q_i$ 在采样整块之前就已确定；串行起草器的 $q_i$ 还依赖前面实际抽出的候选。

把候选接到上下文后，自回归模型的一次因果前向可同时计算

$$
p_i(v)=p_\theta(v\mid s,y_{<i}),\quad i=1,\ldots,m+1.
$$

最后一行用于整块全部通过时的额外抽样。候选使未来位置的输入暂时可用，从而把多次串行生成变成一次并行评分。概率校正沿候选顺序检查，遇到首次拒绝时生成一枚替代 token，并结束本轮。通过的连续前缀进入最终输出。

### 1.3 起草结构

投机解码由“候选分布 $q$”和“目标分布 $p$ 的校正”组成。常见方法主要改变 $q$ 的构造：

| 起草结构 | 候选信息来源 | 主要计算特点 |
|---|---|---|
| 小型自回归模型 | 自身状态与已抽出的候选 | 候选逐项生成 |
| 多项预测 MTP | 共享骨干上的多个预测头或串联预测模块 | 学习多个未来位置；接入目标验证后形成投机推理 |
| 特征辅助起草 | 目标模型提供的中间表示 | 以较小的预测网络补充上下文信息 |
| DFlash 类块起草 | 目标特征与轻量块扩散网络 | 一次前向提出多位置候选 |
| DSpark 类半自回归起草 | 并行表示与轻量串行修正 | 后续候选利用此前实际抽出的 token |
| 本文主线 | 完整共享骨干、真实历史 KV、独立起草注意力 | 一次双向块前向提出候选，一次 AR 前向校正 |

MTP 同时是一类训练目标和模型结构；具体的推理过程取决于预测模块及其验证方式。上述方法可以使用同一套接受与残差校正公式。相关结构分别见 [MTP](https://arxiv.org/abs/2404.19737)、[DFlash](https://arxiv.org/abs/2602.06036) 与 [DSpark](https://arxiv.org/abs/2607.05147)。

## 2. 共享骨干与双向起草注意力

### 2.1 注意力及历史 KV

Transformer 的每层把输入表示 $H$ 投影为查询、键和值：

$$
Q=HW_Q,\qquad K=HW_K,\qquad V=HW_V.
$$

单个注意力头的输出为

$$
\operatorname{Attn}(Q,K,V;M)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt d}+M\right)V.
$$

$d$ 是每个头的维度；掩码 $M$ 在可见位置取 $0$，屏蔽位置取 $-\infty$。查询与键的内积决定各位置的权重，值向量按这些权重求和。实现同时包含逐头 Q/K 归一化和旋转位置编码 RoPE。RoPE 将当前位置的 Q/K 乘以由位置确定的旋转矩阵，编码相对位置关系。

一段已提交历史的各层 $K,V$ 可以缓存。每次续写只计算新输入的 Q/K/V，并读取历史键值。记位置 $a$ 之前的真实 AR 缓存为

$$
C_a=\{K_{<a}^{\mathrm{AR},\ell},
      V_{<a}^{\mathrm{AR},\ell}\}_{\ell=1}^{L},
$$

其中 $L$ 是层数。缓存中的向量由 AR 路径在真实已提交前缀上计算。

### 2.2 两套注意力视图

模型保留 AR 参数 $\theta$，增加一组起草注意力参数 $\phi$：

$$
\phi=\{W_Q^{D,\ell},W_K^{D,\ell},W_V^{D,\ell},W_O^{D,\ell},
       \gamma_Q^{D,\ell},\gamma_K^{D,\ell}\}_{\ell=1}^{L}.
$$

$W_O$ 是注意力输出投影，$\gamma_Q,\gamma_K$ 是逐头归一化的可学习缩放。词嵌入、层归一化、前馈网络和输出头在两条路径间共享。初始化时，各层起草注意力复制相应的 AR 注意力，随后使用独立存储学习。

一次起草前向仍经过全部 Transformer 层。每层使用起草 Q/K/V/O，读取共享的历史 AR KV，并处理当前块的临时起草 KV。模型的存储量为

$$
\text{共享基座参数}+\text{额外起草注意力参数}
+\text{一份持久 AR 历史}+\text{当前块工作区}.
$$

共享参数降低额外权重存储，共享历史降低缓存存储与历史预填充成本。起草耗时仍包含完整层堆栈的前向。

每层的完整计算可以写为

$$
\begin{aligned}
\widetilde H&=H+\operatorname{Attn}_{\phi}(\operatorname{RMSNorm}(H),C),\\
H'&=\widetilde H+
\left[\operatorname{SiLU}(\widehat H W_g)\odot(\widehat H W_u)\right]W_d,
\quad \widehat H=\operatorname{RMSNorm}(\widetilde H).
\end{aligned}
$$

$\odot$ 表示逐元素乘法，$\operatorname{SiLU}(x)=x/(1+e^{-x})$；
RMSNorm 用向量的均方根调整尺度，再乘可学习缩放：

$$
\operatorname{RMSNorm}(h)
=\gamma\odot\frac{h}{\sqrt{d^{-1}\sum_{j=1}^{d}h_j^2+\epsilon}}.
$$

共享前馈权重 $W_g,W_u,W_d$ 在起草学习期间固定，梯度仍经过这些映射传回起草注意力。
冻结参数约束了更新对象，同时保留了完整前向函数对起草参数的导数。

### 2.3 锚点、掩码块与位置对齐

以下从零开始给位置编号。已提交的最新 token 为 $x_a$，称为锚点。轮次开始时：

$$
\text{已提交序列}=x_{0:a},\qquad
\text{持久缓存}=C_a.
$$

因此缓存恰好停在锚点之前。设输入块长为 $B$，起草输入为

$$
z=(x_a,\underbrace{\mathtt{MASK},\ldots,\mathtt{MASK}}_{B-1}),
\qquad \operatorname{pos}(z_j)=a+j.
$$

在第 $\ell$ 层，当前块的起草查询访问

$$
Q_D^\ell,\qquad
K_{\mathrm{all}}^\ell=[K_{<a}^{\mathrm{AR},\ell};K_D^\ell],
\qquad
V_{\mathrm{all}}^\ell=[V_{<a}^{\mathrm{AR},\ell};V_D^\ell].
$$

历史部分全部可见，块内各位置互相可见。双向交互让各个掩码位置交换上下文信息。

输出仍遵循 NTP 的一位偏移：第 $j$ 行预测位置 $a+j+1$。因此使用前 $B-1$ 行：

$$
q_i=\mathcal S(u^D_{i-1}),\qquad
y_i\sim q_i,\qquad i=1,\ldots,m,\quad m=B-1.
$$

| 起草输入行 | 输入位置 | 提供的候选分布 |
|---|---:|---|
| 锚点 $x_a$ | $a$ | $q_1$，预测 $x_{a+1}$ |
| 第一个掩码 | $a+1$ | $q_2$，预测 $x_{a+2}$ |
| 第 $B-2$ 个掩码 | $a+B-2$ | $q_{B-1}$，预测 $x_{a+B-1}$ |
| 最后一个掩码 | $a+B-1$ | 为整块提供双向表示，其输出行在本轮截去 |

最后一个掩码参与其他行的注意力计算；只截去最终输出头中对应的候选行。固定 $\phi$、历史与掩码输入后，基础起草器的整块分布为

$$
q_\phi(y_{1:m}\mid x_{\le a})=\prod_{i=1}^{m}q_{\phi,i}(y_i\mid x_{\le a},z).
$$

各位置共享计算得到的表示，同时按各自的分布独立抽样。验证器则读取实际候选前缀：

$$
\text{验证输入}=(x_a,y_1,\ldots,y_m),\qquad
p_i=p_\theta(\cdot\mid x_{\le a},y_{<i}).
$$

这一结构及其参数映射依据[双视图起草论文](https://arxiv.org/abs/2605.12825)和[参考源码](https://github.com/chiennv2000/orthrus)核对，注意力、缓存、训练与解码循环由本仓独立实现。

## 3. 概率校正及其证明

### 3.1 单个位置的接受与替代

暂时固定当前前缀，并省略位置下标。候选 $Y\sim q$，独立抽取 $U\sim\operatorname{Uniform}[0,1)$。接受条件为

$$
U<\alpha(Y),\qquad
\alpha(v)=\min\left(1,\frac{p(v)}{q(v)}\right).
$$

被实际抽到的候选满足 $q(Y)>0$。一个取值 $v$ 通过接受分支进入输出的概率为

$$
q(v)\alpha(v)=\min(q(v),p(v)).
$$

两种分布的重叠质量为

$$
c=\sum_v\min(p(v),q(v)).
$$

其余概率质量为 $Z=1-c$。利用恒等式
$p(v)=\min(p(v),q(v))+(p(v)-q(v))_+$，可得

$$
Z=\sum_v(p(v)-q(v))_+,
\qquad (b)_+=\max(b,0).
$$

拒绝发生时，从归一化的正残差分布抽取替代 token：

$$
r(v)=\frac{(p(v)-q(v))_+}{Z}.
$$

接受和拒绝两条路径合并后的输出概率为

$$
\begin{aligned}
\Pr(X=v)
&=\Pr(Y=v,\mathrm{接受})+\Pr(\mathrm{拒绝})r(v)\\
&=\min(q(v),p(v))+Z\frac{(p(v)-q(v))_+}{Z}\\
&=p(v).
\end{aligned}
$$

当 $Z=0$ 时，两分布相同，候选必然通过。该质量守恒关系给出了精确投机校正的基础，见[原始投机采样论文](https://proceedings.mlr.press/v202/leviathan23a.html)。

### 3.2 连续前缀与整段输出

假设本轮首次拒绝发生在第 $j$ 个候选。前 $j-1$ 个候选进入输出，第 $j$ 个位置按 $r_j$ 抽样，随后结束本轮。此时已经计算的 $p_j$ 使用上下文 $s,y_{<j}$，其中所有候选都已接受，因此正好是该输出位置所需的目标分布。

若 $m$ 个候选全部通过，使用最后一行 $p_{m+1}$ 再抽取一枚 token。把单位置的质量守恒结论依次应用于每个输出前缀，可得

$$
\Pr(X_{1:n}=x_{1:n}\mid s)
=\prod_{i=1}^{n}p_\theta(x_i\mid s,x_{<i}).
$$

证明中的归纳条件是：在给定到达当前位置之前的信息后，当前候选按照保存的 $q_i$ 抽样，接受随机数与候选抽样独立，校正使用对应前缀上的 $p_i$。

在线参数同样可以依赖已经完成的轮次。记第 $t$ 轮开始时的信息为 $\mathcal F_t$，起草参数为 $\phi_t=F(\mathcal F_t)$。条件于 $\mathcal F_t$，本轮是一个固定 $q_{\phi_t}$ 的普通校正过程；再对 $\mathcal F_t$ 取平均，上式仍成立。实现为每轮保存候选概率快照，完成校正与提交后发布 $\phi_{t+1}$。

贪心模式把 $p,q$ 视为集中于最大分数 token 的点分布，接受条件化为两者取值相等。这个结论基于给定的数值 logits；低精度矩阵运算的布局差异需要通过独立数值审计衡量。

### 3.3 接受率与总变差

总变差距离定义为

$$
D_{\mathrm{TV}}(p,q)=\frac12\sum_v|p(v)-q(v)|.
$$

由
$\min(a,b)=\frac12(a+b-|a-b|)$ 和 $\sum p=\sum q=1$，得到

$$
\boxed{\Pr(\mathrm{接受}\mid\text{当前前缀})
=1-D_{\mathrm{TV}}(p,q).}
$$

这说明词表分布的重叠质量直接决定当前位置的平均接受率。对于某一个已抽出的候选，其接受概率是 $\alpha(Y)$；平均所有可能候选后才得到 $1-D_{\mathrm{TV}}$。

### 3.4 整块张量执行

给定候选和所有验证行，接受阈值可同时计算：

$$
a_i=\min(1,p_i(y_i)/q_i(y_i)),\qquad
A=\min\bigl(\{i-1:U_i\ge a_i\}\cup\{m\}\bigr).
$$

$A$ 是连续接受数。张量执行通过一次索引归约找到 $A$，选择 $p_{A+1}$，并在需要时减去对应 $q_{A+1}$。索引 $i$ 从一开始；代码数组从零开始。

替代抽样采用指数竞争。对每个词表项独立抽取 $E_v\sim\operatorname{Exp}(1)$，令

$$
X=\arg\min_{v:q(v)>0}\frac{E_v}{q(v)}
=\arg\max_v\frac{q(v)}{E_v}.
$$

随机变量 $E_v/q(v)$ 的指数速率为 $q(v)$。某个 $v$ 最先到达的概率为

$$
\int_0^\infty q(v)e^{-q(v)t}
       \prod_{w\ne v}e^{-q(w)t}\,dt
=q(v)\int_0^\infty e^{-t}\,dt=q(v).
$$

同样的竞争可直接使用未归一化的正残差，因为公共归一化常数保持排序。张量路径把接受数、替代 token 和有效性标记合并传回主机，减少逐位置同步。普通张量执行和可选 GPU 图共享此映射，随机数在每次执行前重新生成，返回的概率快照拥有独立存储。

## 4. 持久缓存的不变量

### 4.1 轮次开始与结束

轮次开始时，锚点位于 $a$，持久缓存为 $C_a$。起草读取这份缓存，临时块 KV 随本次前向结束释放。验证器读取 $(x_a,y_1,\ldots,y_m)$，产生长度为 $a+m+1$ 的临时 AR 缓存。

设本轮接受 $A$ 个候选，并提交一枚替代或额外 token $c$。新的已提交序列是

$$
(x_{0:a},y_1,\ldots,y_A,c).
$$

新锚点为 $c$，位置为 $a+A+1$。因此保留验证缓存的前 $a+A+1$ 个位置：

$$
C_{\mathrm{next}}
=C_{\mathrm{verify}}[\,:\,a+A+1].
$$

这份缓存包含旧锚点和所有接受候选，恰好停在新锚点之前。请求达到 EOS 或输出预算时，提交在对应位置结束，缓存按实际提交长度裁剪。

### 4.2 归纳证明

初始 prefill 在给定提示上运行 AR，缓存所有提示 token，并从最后一行抽取第一枚输出。该输出成为锚点，缓存停在它之前，不变量成立。

假设某轮开始时缓存等于真实前缀的 AR 缓存。验证器使用因果注意力，所以位置 $a+i$ 的 KV 只依赖它及其之前的输入。对已经接受的前缀，这些输入与最终序列逐项相同。因此裁剪后保留的每层 KV 等于在该真实前缀上重新运行 AR 得到的 KV。下一轮的不变量随之成立。

在线训练只发布起草注意力参数。历史缓存由冻结 AR 路径产生，故跨轮更新后上述归纳条件继续成立。共享历史的依据是这条缓存不变量。

## 5. 起草分支的离线蒸馏

### 5.1 随机锚点与多块掩码

训练样本为已经给定的干净序列 $x_{0:T-1}$。从满足 $0\le a_b\le T-B$ 的位置抽取若干锚点 $a_1,\ldots,a_K$。每个锚点形成一个输入块：

$$
z^{(b)}=(x_{a_b},\mathtt{MASK},\ldots,\mathtt{MASK}),
\qquad \operatorname{pos}(z_j^{(b)})=a_b+j.
$$

干净序列先经过冻结 AR 路径，产生所有层的历史 KV 与教师表示。所有起草块拼成长度 $KB$ 的序列，通过显式掩码隔离信息。

设起草查询属于块 $b$，键来自干净序列位置 $k$ 或某个起草块 $b'$。可见条件为

$$
\operatorname{visible}(b,k,b')
=
\begin{cases}
k<a_b,& \text{键来自干净序列},\\
b'=b,& \text{键来自起草序列}.
\end{cases}
$$

每个起草块读取锚点之前的真实 AR 历史及本块全部位置。RoPE 位置使用原序列中的 $a_b+j$，而非拼接数组中的偏移。因果教师保证历史位置的表示只依赖对应的真实前缀；块隔离保证多块打包与各块单独起草具有相同的信息条件。

### 5.2 教师行、学生行与完整分布

第 $b$ 个块、第 $j$ 行的学生 logits 对应位置 $a_b+j+1$，其教师来自干净序列的第 $a_b+j$ 行：

$$
\begin{aligned}
p_{b,j}&=\operatorname{softmax}
 \bigl(u^{\mathrm{AR}}_{a_b+j}\bigr)
 =p_\theta(\cdot\mid x_{\le a_b+j}),\\
q_{b,j}&=\operatorname{softmax}(u^{D}_{b,j}),
\qquad j=0,\ldots,B-2.
\end{aligned}
$$

离线蒸馏最小化完整词表上的前向 KL：

$$
\mathcal L(\phi)
=\frac{1}{K(B-1)}
 \sum_{b=1}^{K}\sum_{j=0}^{B-2}
 \sum_v p_{b,j}(v)\log\frac{p_{b,j}(v)}{q_{\phi,b,j}(v)}.
$$

批量训练时再对样本取平均。教师项停止梯度，只更新起草注意力参数 $\phi$。实现按若干输出行分块计算完整词表损失，并在反向时重算输出头中间量，以限制同时驻留的词表维工作区。

本仓的训练目标对应论文中的完整分布蒸馏。核对的公开训练入口采用真实 token 的交叉熵监督，
二者沿用相同的随机锚点、双视图掩码与下一项行对齐。训练入口与损失的选择在各自实现中明确区分。

设某行学生 logits 为 $u$。由于
$\partial\log q(v)/\partial u(w)=\mathbf1[v=w]-q(w)$，有

$$
\frac{\partial D_{\mathrm{KL}}(p\Vert q)}{\partial u(w)}
=-\sum_v p(v)(\mathbf1[v=w]-q(w))
=q(w)-p(w).
$$

对 logits 直接做梯度下降时，教师概率高于学生的词表位置获得提高分数的方向，反之降低。离线入口同时保存优化器、随机数、数据顺序和学习率调度，使中断恢复对应连续训练中的同一更新序列。

### 5.3 并行起草的条件信息

对第 $i$ 个候选，AR 教师已经观察到该位置之前的真实 token；起草器只观察到块起点和掩码表示。因此相同起草输入 $s$ 可以对应多个教师条件分布。记其平均分布为 $\bar p=\mathbb E[p\mid s]$，则

$$
\begin{aligned}
\mathbb E[D_{\mathrm{KL}}(p\Vert q)\mid s]
&=\mathbb E[D_{\mathrm{KL}}(p\Vert\bar p)\mid s]
  +D_{\mathrm{KL}}(\bar p\Vert q).
\end{aligned}
$$

证明可将 $\log(p/q)$ 拆成 $\log(p/\bar p)+\log(\bar p/q)$，再对第二项中的 $p$ 取条件期望。第一项由教师在相同起草输入下的变化决定，第二项衡量起草器对平均分布的拟合。

由此，独立并行预测的 KL 最优解是可见信息下的条件平均 $\bar p$。块内多种合理续写越分散，后续位置的条件信息差距越大。这也给出了半自回归修正的研究动机；本仓相应对照实现集中于 [ablation](../ablation/README.md)。

## 6. 推理期间的起草注意力续训

### 6.1 实际验证反馈

在线版本从已经加载的预训练起草参数开始，参数与优化器状态跨请求保留。
利用验证概率进行在线蒸馏的研究见 [Online Speculative Decoding](https://arxiv.org/abs/2310.07177)
与 [Test-Time Speculation](https://arxiv.org/abs/2605.09329)。以下推导给出共享缓存与后段重放的具体实现条件。

本轮候选为 $y_{1:m}$，连续接受数为 $A$。若发生拒绝，前 $A+1$ 个教师行具有有效的已接受前缀；其中最后一行提供拒绝位置的目标分布。若整块通过，则全部 $m$ 个候选行有效。因此有效监督行数为

$$
n=\min(A+1,m).
$$

EOS 和输出预算会进一步缩短实际到达的位置范围。每个反馈块保存起草输入、真实 AR 历史、有效教师 logits 与起草后段的输入边界。概率校正使用本轮实际采样时保存的 $q$；反馈训练在校正完成后执行。

### 6.2 冻结边界与完整块重放

把起草层划分为冻结前段 $F$ 和可更新的注意力后段 $G_\phi$。一次起草中，后段入口表示为

$$
H_\ast=F(z,C),\qquad
u_\phi=\operatorname{Head}(G_\phi(H_\ast,C)).
$$

$F$、共享 AR 参数与缓存均固定，故 $\partial H_\ast/\partial\phi=0$。保存 $H_\ast$ 后，重放后段计算得到的梯度为

$$
\nabla_\phi\mathcal L
=\left(\frac{\partial u_\phi}{\partial\phi}\right)^\top
 \nabla_{u_\phi}\mathcal L,
$$

与对同一个输入运行完整起草网络、只求后段参数梯度相同。

后段重放保留 $B$ 个输入位置及其原有位置编码和掩码，再选取前 $n$ 行计算监督损失。双向注意力使有效行依赖块内后续掩码位置，完整块形状是梯度等价的条件。数值检查分别验证完整前向与后段重放的 logits、解析梯度及有限差分。

训练时输出头只计算有效监督行，以节省词表维计算；逐元素执行审计使用与真实起草相同的完整输出头形状。

只保存参与后段计算的各层历史 KV。缓存采用只读张量视图，后续轮次通过拼接和裁剪生成新缓存；保存的边界、输入和教师 logits 则拥有独立快照。

### 6.3 更新窗口与参数发布

设每 $s$ 个有效轮次更新一次，每次重放最近 $r$ 个反馈块，其中 $1\le r\le s$。
用 $D$ 表示所选的分布距离，窗口中的损失按有效位置数归一化：

$$
\mathcal L_t(\phi)=
\frac{\sum_{b\in\mathcal R_t}\sum_{i=1}^{n_b}
D(p_{b,i},q_{\phi,b,i})}
{\sum_{b\in\mathcal R_t}n_b}.
$$

只有即将参与更新的轮次保存冻结边界，以控制复制与显存开销。每个请求结束时释放重放数据，保留学习参数、优化器状态和更新计数。

训练维护 FP32 主权重 $\phi^{32}$，前向使用对应的执行精度：

$$
\phi^{\mathrm{exec}}=\operatorname{cast}(\phi^{32}).
$$

反向通过转换后的主权重计算，梯度裁剪后由 AdamW 更新。更新完成后，把主权重复制到起草执行张量，供下一轮使用。AR 与共享参数始终固定。这使低精度推理中的小幅学习增量可以在 FP32 中累积。

### 6.4 分布目标与接受目标

当前续训支持完整分布 KL 和 TV。前者的 logits 梯度为 $q-p$；后者由

$$
D_{\mathrm{TV}}(p,q)=\frac12\sum_v|q(v)-p(v)|
=1-\sum_v\min(p(v),q(v))
$$

直接联系平均接受率。固定当前位置的历史，减小 TV 等价于提高这一位置的平均接受概率。
这条等式对实际送入概率校正的 $p,q$ 成立。温度一、完整词表的设置使训练与校正使用同一对分布。

TV 的梯度也可以直接写出。令 $g(v)=\operatorname{sign}(q(v)-p(v))$，
相等处选取次梯度 $g(v)=0$。把 softmax 导数
$\partial q(v)/\partial u(w)=q(v)(\mathbf1[v=w]-q(w))$ 代入，得到

$$
\begin{aligned}
\frac{\partial D_{\mathrm{TV}}}{\partial u(w)}
&=\frac12\sum_v g(v)q(v)(\mathbf1[v=w]-q(w))\\
&=\frac12q(w)\left[g(w)-\sum_vq(v)g(v)\right].
\end{aligned}
$$

当 $q(w)<p(w)$ 时，这个导数小于或等于零，对 logits 直接做梯度下降会提高该位置的分数；
$q(w)>p(w)$ 时方向相反。所有分数通过 softmax 的归一化耦合，词表总概率始终为一。
完整词表上的教师概率同时参与损失和梯度。

在线窗口固定已经获得的上下文与教师分布，把这次反馈作为一次局部拟合问题。
下一轮更新后的起草器会产生新的候选与反馈。整轮接受长度还受“能否到达后续位置”影响，
其精确表达见第 7.1 节；逐行 TV 为这个整体目标提供易计算的局部监督。

KL 与 TV 之间还有一个定量关系。令 $E=\{v:p(v)\ge q(v)\}$，则
$\delta=p(E)-q(E)=D_{\mathrm{TV}}(p,q)$。把词表合并为 $E$ 及其补集，log-sum 不等式给出

$$
D_{\mathrm{KL}}(p\Vert q)
\ge d(p(E)\Vert q(E)),
$$

其中
$d(a\Vert b)=a\log(a/b)+(1-a)\log((1-a)/(1-b))$。
固定 $b$，该函数在 $a=b$ 处取零且一阶导为零，二阶导满足

$$
\frac{\partial^2d}{\partial a^2}
=\frac{1}{a(1-a)}\ge4.
$$

对二阶导积分两次，得到 $d(a\Vert b)\ge2(a-b)^2$，于是

$$
D_{\mathrm{TV}}(p,q)\le
\sqrt{\frac12D_{\mathrm{KL}}(p\Vert q)}.
$$

结合第 3 节：

$$
\Pr(\mathrm{接受}\mid s)
\ge1-\sqrt{\frac12D_{\mathrm{KL}}(p\Vert q)}.
$$

此界要求左右两侧使用同一对分布。KL 给出接受率的下界，TV 给出该条件接受率的确切值。
当前完整分布训练以温度一的原始 softmax 为对象；其他温度和截断规则定义
$\mathcal S(u^{\mathrm{AR}}),\mathcal S(u^D)$。因此评估同时记录原始 KL、原始 TV 和实际采样 TV，
并以配对生成实验判断分布改善与学习开销的合计效果。

### 6.5 学习界的适用条件

对固定反馈分布，设平均损失 $F(\phi)$ 下界为 $F_\ast$，梯度满足 $L$-光滑条件。
本节适用于满足该条件的平滑损失；TV 的绝对值拐点使用上一节的次梯度。
若梯度估计 $g_t$ 无偏、方差至多 $\sigma^2$，并使用
$\phi_{t+1}=\phi_t-\eta g_t$，其中 $\eta\le1/L$，则光滑性给出

$$
\mathbb E[F(\phi_{t+1})\mid\phi_t]
\le F(\phi_t)-\frac{\eta}{2}\|\nabla F(\phi_t)\|^2
+\frac{L\eta^2\sigma^2}{2}.
$$

对 $t=0,\ldots,T-1$ 求和，整理得

$$
\frac1T\sum_{t=0}^{T-1}
\mathbb E\|\nabla F(\phi_t)\|^2
\le
\frac{2(F(\phi_0)-F_\ast)}{\eta T}
+L\eta\sigma^2.
$$

它描述固定分布、随机梯度更新下接近平稳点的条件。实际程序使用 AdamW、裁剪、有限窗口和随起草参数变化的验证轨迹，评估对象是这个具体过程。重放梯度精确性由第 6.2 节保证；最终接受质量与吞吐由第 7 节的端到端测量检验。两种结论分别对应局部优化计算和整体生成性能。

## 7. 接受长度、时间成本与实验判据

### 7.1 连续接受长度

设 $A$ 是一轮连续接受的候选数，取值为 $0,\ldots,m$。对任意这样的整数随机变量，

$$
A=\sum_{k=1}^{m}\mathbf1[A\ge k],
\qquad
\mathbb E[A]=\sum_{k=1}^{m}\Pr(A\ge k).
$$

固定当前历史，记前 $k$ 枚全部通过的概率为 $S_k$。把所有可能候选前缀求和，得到精确表达：

$$
S_k=
\sum_{y_{1:k}}
\prod_{i=1}^{k}
\min\bigl(p_i(y_i\mid s,y_{<i}),q_i(y_i\mid s,y_{<i})\bigr).
$$

基础并行起草的 $q_i$ 对块内已采样前缀保持固定；上式同时适用于带条件依赖的起草器。定义

$$
\bar c_i=
\mathbb E\left[
1-D_{\mathrm{TV}}(p_i,q_i)\mid A\ge i-1
\right],
$$

其中期望针对已经存活到第 $i$ 个位置的候选前缀分布，则
$S_i=S_{i-1}\bar c_i$，$S_0=1$。单条轨迹上的各位置 TV 可以作为诊断量；整轮期望接受长度由前缀存活概率决定。

如果用常数 $c$ 近似所有条件接受率，则

$$
\mathbb E[A]\approx\sum_{k=1}^{m}c^k,
\qquad
\mathbb E[N]\approx1+\sum_{k=1}^{m}c^k
=\frac{1-c^{m+1}}{1-c}.
$$

$N=A+1$ 是一轮提交的 token 数，包含替代或额外 token。$c=1$ 时取极限 $N=m+1=B$。EOS 和请求末尾预算另作边界处理。

### 7.2 净吞吐条件

记普通 AR 每 token 耗时为 $C_{\mathrm{AR}}$，一轮起草、验证和采样控制的总耗时为

$$
C_0=C_D+C_V+C_S.
$$

忽略有限请求的启动和末尾效应，在稳定轮次统计下，吞吐近似为

$$
\operatorname{TPS}_{\mathrm{spec}}
=\frac{\mathbb E[N]}{\mathbb E[C_0]},
\qquad
\operatorname{Speedup}
=\frac{C_{\mathrm{AR}}\mathbb E[N]}{\mathbb E[C_0]}.
$$

这里使用总产出除以总时间，对应长期“每轮平均产出／每轮平均耗时”。若每 $s$ 轮更新一次、每次更新耗时 $C_U$、平均反馈开销为 $C_F$，则

$$
\operatorname{TPS}_{\mathrm{online}}
\approx
\frac{\mathbb E[N_{\mathrm{online}}]}
{\mathbb E[C_0^{\mathrm{online}}]+C_F+C_U/s}.
$$

在其他轮次成本近似相同的简化条件下，在线版本相对固定版本获益需要

$$
\frac{\mathbb E[N_{\mathrm{online}}]}{\mathbb E[N_{\mathrm{fixed}}]}
>
1+\frac{C_F+C_U/s}{\mathbb E[C_0]}.
$$

这把接受长度的改善与付出的学习时间放在同一尺度上。实际实验以完整请求 TPS 判断净收益，并分别记录反馈、更新、初始化和独立预学习成本。

### 7.3 配对对照与数值验证

主实验采用五组：AR、预训练起点固定推理、该起点在线续训、独立学习后固定推理、同一学习状态继续在线更新。后两组共享初始学习参数和优化器状态，比较学习后保留参数与继续学习的差异。

同一问题交错运行各组，保持模型精度、提示模板、采样规则、EOS 规则和输出预算一致。
每条重复流从相同起始学习状态重新开始，单独打乱问题顺序；问题对应的随机种子由问题编号和重复编号确定。
记录每条流的固定与在线 TPS，可分别观察顺序变化及生成随机性带来的波动。

每组 TPS 为输出 token 总数除以请求总时间。置信区间以问题为簇配对重采样，
保留同一个问题在所有重复中的记录。它衡量给定学习轨迹中的问题间波动；
各条独立学习流的增益同时报告，以补充跨请求更新所产生的序列相关性。

数学与实现的对应检查包括：

- 枚举小词表的接受、拒绝和替代分支，核对最终概率质量。
- 比较逐 token AR 与并行验证的 logits、历史 KV 和贪心输出。
- 比较多锚点打包训练与逐块训练的输出、损失和梯度。
- 比较完整块前向与后段重放，并用有限差分检查更新参数。
- 比较普通张量与 GPU 图的候选、接受计数、输出和前向次数。
- 比较连续训练与断点恢复的参数、优化器、数据顺序及随机状态。
- 在相同实际候选前缀上比较学习前后分布，并单独测量持续更新轨迹的净 TPS。

## 8. 实现索引

| 数学内容 | 主线模块 |
|---|---|
| 双视图层、共享参数、历史 KV、后段前向 | `parallel/backbone.py` |
| 锚点输入与候选行对齐 | `parallel/branches.py` |
| 轮次循环、提交和缓存裁剪 | `parallel/generation.py`、`state.py` |
| 采样变换、接受与正残差 | `sampling.py`、`sampling_execution.py` |
| 随机锚点、块掩码、分块完整分布损失 | `parallel/training.py`、`losses.py` |
| 数据流、优化器调度与训练恢复 | `parallel/fitting.py` |
| 反馈窗口、FP32 主权重、在线发布 | `feedback.py`、`parallel/feedback.py`、`parallel/online.py` |
| 同前缀审计与配对区间 | `parallel/audit.py`、`measurement.py` |
| 权重导入与参考完整性检查 | `parallel/weights.py`、`reference.py` |
| 固定评测、在线对照、训练入口 | `commands/evaluate.py`、`commands/continue_training.py`、`commands/fit.py` |

入口与命令见 [RUNNING.md](RUNNING.md)。对照分支的实现、推导摘要与取舍集中在 [ablation/README.md](../ablation/README.md)。默认软件包与测试仅覆盖本报告主线。
