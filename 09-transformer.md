# Transformer

> 论文原文：https://arxiv.org/abs/1706.03762

## 架构图

```
输入文本 ──► embedding ──► + 位置编码
                              │
                              ▼
    ┌────────────── Encoder Block × N ──────────────┐
    │  ① 多头自注意力 ──► ② Add & Norm               │
    │  ③ FFN         ──► ④ Add & Norm               │
    └───────────────────────┬───────────────────────┘
                            │ 编码器输出
                            ├────────────► 给解码器 ⑦ 当 K、V
                            │
已生成的词 ──► embedding ──► + 位置编码
                            │
                            ▼
    ┌────────────── Decoder Block × N ──────────────┐
    │  ⑤ 带 mask 的多头自注意力 ──► ⑥ Add & Norm     │
    │  ⑦ 交叉注意力（Q 来自解码器，K、V 来自编码器）  │
    │                        ──► ⑧ Add & Norm       │
    │  ⑨ FFN                 ──► ⑩ Add & Norm       │
    └───────────────────────┬───────────────────────┘
                            ▼
                     Linear ──► Softmax
                            ▼
                         输出概率
```

## 输入：embedding 与 position encoding

1. embedding（词嵌入，向量嵌入）：就是把文字，图片这些转换成电脑可以读懂的数字向量
2. position encoding（位置编码：明确顺序）：处理 token 不知道谁先谁后的问题，加入了 position encoding

$$
PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\qquad
PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

- 第 **偶数**维（2i）用 sin
- 第 **奇数**维（2i+1）用 cos
- 每一对 (2i, 2i+1) 共享同一个频率

| 符号 | 含义 | 说明 |
| --- | --- | --- |
| `pos` | token 在序列里的位置 | 0, 1, 2 … L−1 |
| `i` | 维度对的编号 | 范围 **0 ~ d_model/2 − 1**，不是 d_model |
| `d_model` | 词嵌入矩阵的大小 | 例：64 |

## 零件一：attention

> 注意力机制最核心的思想：让当前 token 判断其他 token 对自己到底有多重要，重点读取重要的信息

1. **Query** 搜索关键词（找什么）
2. **key** 每个 token 提供什么信息支持查询，专门拿来和 Q 去做可以匹配的向量（有什么）
   > Q 点积 K（注意力分数）：匹配度打分 —— 通过 softmax 把得出的结果转化成权重

$$
Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**为什么要除以sqrt{d_k}**：防止softmax饱和导致梯度消失


3. **Value** 负责保存真正有需要的信息（有啥信息）
   > 用得到的权重 * value 判断

### 角色关系

```
      Q                    K                     V
  ─────────            ─────────            ─────────
   我要找什么             我是什么             我的内容
   (Query 查询)          (Key 键)            (Value 值)

       │                    │                     │
       └─────── 比对 ───────┘                     │
                  │                               │
           相关度有多高？                          │
                  │                               │
            ┌─────┴─────┐                         │
            ↓           ↓                         │
        高相关       低相关                        │
        权重大       权重小                        │
            │           │                         │
            └─────┬─────┘                         │
                  ↓                               │
           得到一组权重 ───────────────────────────┘
                  │
                  ↓
          按权重把 V 混合起来
                  │
                  ↓
               输出
```

### 对投影的理解

> 投影就是一层 nn.Linear，即不带激活的线性变换。把一个 token 向量，用不同的权重矩阵翻译到不同的空间，nn.Linear(a, b) 做的事是 y = x @ W.T + b，只改最后一维，形状 [B, n, a] → [B, n, b]

1. 为什么要投影？
   > 如果省掉投影，让 Q=K=V=x，注意力分数就是 x*x，并没有什么比对价值
   > 加上 W_q,W_k,W_v，网络可以决定：提问时要什么，啥样的线索可以匹配的上，被匹配后应该拿出哪些信息
   > 投影 W_o（代码 out_proj）就是把各个结果融合在一起形成一份输出
2. 多头里面的投影
   > d_model=64,h=4 → d_k=16：把 64 维空间切成 4 个 16 维子空间，每个头只在自己那 16 维里算注意力
   > 代码里面 view(B,n,h,d_k).transpose(1,2) 就是切；transpose(1,2).contiguous().view(B,n,d_model) 就是拼

### 多头并行计算 Multi-Head Attention

> "头"指的是独立的一套 Query,Key,Value 权重矩阵，实现多角度全方位的理解

| 字母 | 英文全称 | 含义 | 人话 | 典型值 |
| --- | --- | --- | --- | --- |
| `B` | batch size | 批大小 | 一次喂几份数据 | 2 |
| `n` | sequence length | 序列长度 | 一句话有几个 token | 10 |
| `h` | num_heads | 头数 | 分成几个头看 | 4 |
| `d_k` | head dim | 每头维度 | 每个头分到多少维 | 16 |
| `d_model` | model dim | 模型维度 | 每个 token 的总向量长度 | 64 |

做H组独立的注意力机制
$$
\mathrm{head}_i = \mathrm{Attention}\left(QW_i^{Q},\; KW_i^{K},\; VW_i^{V}\right)
$$

接线并且线性融合回模型维度
$$
\mathrm{MultiHead}(Q,K,V) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)\,W^{O}
$$

### 掩码注意力机制 Masked Multi-head-attention
1. 为什么需要填充：在处理批次时，Transformer的输入要求**同一个batch**中所有句子长度一致，以便可以在GPU上并行计算
2. 做注意力时把"不该看的位置"的分数压成 −∞，softmax 之后这些位置权重约等于 0，等于物理上看不见。

### 交叉注意力机制

| 对比项 | 自注意力 | 交叉注意力 |
| --- | --- | --- |
| Q 来自 | 解码器自己 | 解码器自己（我要找什么） |
| K / V 来自 | 解码器自己 | 编码器的 memory（原文有什么） |
| 分数矩阵形状 | `[n, n]` 方阵 | `[n_tgt, n_src]` 矩形 |
| 需要哪种 mask | causal + padding | 只要 padding（没有"未来"） |

### 层归一化 Norm

公式LN：


$$
\begin{aligned}
\mu &= \frac{1}{d}\sum_{i=1}^{d} x_i
\qquad
\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2 \\[6pt]
\mathrm{LN}(x) &= \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
\end{aligned}
$$

**对比**：现在更常用的RMSNorm(没有均值的那一步，只保留均方根)

### RMSNorm 公式

$$
\mathrm{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}
$$

$$
\mathrm{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\mathrm{RMS}(x)^2 + \varepsilon}}
$$

> 为什么会有RMS这个公式的出现？ 优化之后有什么效果

答案：RMSNorm对于均值（中心化）和标准差（缩放）分别做消融实验，得到的结论就是LN真正起作用的是缩放，所以去掉了均值

## 零件二：FFN

> 前馈神经网络：负责把每个 Token 的信息组合起来，进行分析研判，简单说就是思考

## 梯度消失与残差连接

**梯度消失：有这两个零件叠加之后，层数会变多，在损失函数作用下，反向传播，梯度一层一层往回传，层数一多，传到前面的微乎其微，对于前面而言没有起到任何作用**

> 解决梯度消失的办法：残差连接恒等映射公式

### 残差连接搭配归一化模块 Add & Norm

## 解码器：把高维向量转化成字

1. 解码器生成机制规则：只能看到自己和前面的 token
2. 怎么做到这种效果：**mask** 计算注意力分数的时候，直接把后面的分数转换成一个极小的数，经过 softmax 权重几乎为 0，就看不到后面的 token 了
3. 解码器怎么知道原文在说什么：解码器一边解码一边回看编码器，拿自己的 Q 去查询编码器输出的 k 和 v

## Transformer推理和训练过程

### 推理过程：简单来说就是编码器跑一次，解码器吐出一个字

### 训练过程： 简单来说就是抄答案改错，token进行逐个对比，反向传播更新权重

#### 训练过程文字描述
1. 分词→ 加 <BOS> → 补 <PAD>
2. Encoder:原句整句并行算一遍，输出memory[B,n_src,d_model]
3. Decoder:目标句子右移一位喂进去，每个位置都是看着左边去猜自己
4. 损失:???输出过nn.Linear(d_model,vocab_size)得到logits,逐 token 交叉熵？？？？,<PAD>用ignore_index跳过
5. 反向：梯度裁剪+Adam学习率

#### 代码实现
```
model.train()
for src,tgt in loader:
   logits=model(src,tgt[:,:-1])  #右移输入
   loss=F.cross_entropy(logits.reshape(-1, V), tgt[:, 1:].reshape(-1),
                           ignore_index=PAD, label_smoothing=0.1)
   loss.backward()
   clip_grad_norm_(model.parameters(), 1.0)
   opt.step();
   opt.zero_grad()

```
