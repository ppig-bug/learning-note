# 01 强化学习:基本概念、MDP 与贝尔曼公式

> 入门了解:强化学习的最终目的就是要求解最优策略。

## 基本概念

辅助理解的一个模型:grid-world example(网格世界示例),如下:

| s1 | s2 | s3 |
|----|----|----|
| s4 | s5 | **<span style="background-color:#ffbc00">s6</span>** |
| **<span style="background-color:#ffbc00">s7</span>** | s8 | **<span style="background-color:#40e0f0">s9</span>** |

> 图例:黄色 = Forbidden area(禁止区域),蓝色 = 目标区域

### 1. State(状态)

agent 相对于环境的状态,在 grid 中就是 location,即 s1、s2 等。

### 2. State Space(状态空间)

本质就是一个集合:$S = \{s_i\}$。

### 3. Action(动作)

每个状态对应 5 个可能的动作:

- $a_1$:向上
- $a_2$:向右
- $a_3$:向下
- $a_4$:向左
- $a_5$:不动

### 4. Action Space of a State(状态的动作空间)

同理,也是一个集合。

### 5. State Transition(状态转移)

实际上定义了 agent 和环境的交互行为。举例:$s_1$ 通过 $a_2$ 到达 $s_2$;$s_1$ 通过 $a_1$ 仍在 $s_1$(原地不动)。

### 6. Forbidden Area(禁止区域,黄色)

- situation 1:accessible but with penalty(可进入但有惩罚)
- situation 2:inaccessible(不可进入)

### 7. State Transition Probability(状态转移概率)

用条件概率描述 state transition:

$$p(s_2|s_1,a_2)=1,\qquad p(s_i|s_1,a_2)=0 \quad (i \neq 2)$$

### 8. Policy(策略)

在某个状态下采取怎样的 action。

### 9. Mathematical Representation(数学表示,确定性策略)

> 注:强化学习中 $\pi$ 统一指的是策略。

$$\pi(a_1|s_1)=0,\quad \pi(a_2|s_1)=1,\quad \pi(a_3|s_1)=0,\quad \pi(a_4|s_1)=0,\quad \pi(a_5|s_1)=0$$

### 10. Stochastic Policies(随机策略,不确定性)

$$\pi(a_1|s_1)=0,\quad \pi(a_2|s_1)=0.5,\quad \pi(a_3|s_1)=0.5,\quad \pi(a_4|s_1)=0,\quad \pi(a_5|s_1)=0$$

### 11. Reward(奖励)

一个标量,在 agent 采取动作后得到一个数。正数即鼓励,负数即惩罚,为 0 相当于无奖励。

$$p(r=-1|s_1,a_1)=1,\qquad p(r \neq -1|s_1,a_1)=0$$

### 12. Trajectory(轨迹)

state-action-reward 链。

### 13. Return(回报)

trajectory 中所有 reward 相加。

### 14. Discounted Return(折扣回报)

为解决 return 无限累加得到无穷值的问题,引入折扣率 $\gamma$,其中 $0 \le \gamma \le 1$:

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

- $\gamma \to 0$:return 越依赖于前面的奖励
- $\gamma \to 1$:return 越依赖后面的奖励

### 15. Episode(回合/片段)

一个有限步的 trajectory。

## Markov Decision Process(MDP,马尔可夫决策过程)

### 1. Key Elements of MDP(MDP 的关键要素)

- sets:state、action、reward
- Probability distribution:state transition probability、reward probability
- Policy:$\pi(a|s)$

### 2. Markov Property(马尔可夫性质)

$$p(s_{t+1}|a_{t+1},s_t,\ldots,a_1,s_0)=p(s_{t+1}|a_{t+1},s_t)$$

$$p(r_{t+1}|a_{t+1},s_t,\ldots,a_1,s_0)=p(r_{t+1}|a_{t+1},s_t)$$

## 贝尔曼公式
vi denote the return obtained starting from si(i=1,2,3,4)
- v1=r1+γr2+γ<sup>2</sup>r3+...
- v2=r2+γr3+γ<sup>2</sup>r4+...
- v3=r3+γr4+γ<sup>2</sup>r1+...
- v4=r4+γr1+γ<sup>2</sup>r2+...

把γ提出来：v1=r1+γv2 由此可以得到一个结论：一个状态的value依赖于其他状态的value
> 将公式用矩阵表示出来得到：

    Bellman equation: v=r+γPv

### State value
本质上就是
> $$G_t的一个期望值或者平均值

> $$v_Π(s)=E[Gt|St=s]

区别：Return是针对单个trajectory得到的return；State value 是针对多个trajectory得到的return再求平均值


### 贝尔曼期望方程（状态价值）完整推导

回报定义：
$$G_t = R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\dots = R_{t+1}+\gamma G_{t+1}$$

状态价值函数定义：
$$v_\pi(s) = \mathbb{E}\big[G_t \mid S_t = s\big]$$

代入回报递推关系：

$$
\begin{aligned}
v_\pi(s)
&= \mathbb{E}\big[R_{t+1}+\gamma G_{t+1} \mid S_t = s\big] \\
&= \mathbb{E}\big[R_{t+1} \mid S_t = s\big]
+\gamma\,\mathbb{E}\big[G_{t+1} \mid S_t = s\big]
\end{aligned}
$$

#### ① 展开即时奖励期望
$$\begin{aligned}
\mathbb{E}\big[R_{t+1}\mid S_t = s\big]
&=\sum_{a}\pi(a|s)\; \mathbb{E}\big[R_{t+1}\mid S_t=s,\,A_t=a\big] \\
&=\sum_{a}\pi(a|s)\left(\sum_{r} p(r\mid s,a)\, r\right)
\end{aligned}$$

#### ② 展开未来回报期望
$$
\begin{aligned}
\mathbb{E}\big[G_{t+1}\mid S_t = s\big]
&=\sum_{s'}\mathbb{E}\big[G_{t+1}\mid S_t=s,\;S_{t+1}=s'\big]\, p(s'\mid s) \\
&=\sum_{s'}\mathbb{E}\big[G_{t+1}\mid S_{t+1}=s'\big]\, p(s'\mid s) \\
&=\sum_{s'} v_\pi(s')\; p(s'\mid s)
\end{aligned}
$$

状态边缘转移概率全概率展开：
$$p(s'|s)=\sum_{a} p(s'\mid s,a)\,\pi(a|s)$$

$$
\mathbb{E}\big[G_{t+1}\mid S_t = s\big]
=\sum_{s'} v_\pi(s') \sum_{a} p(s'\mid s,a)\,\pi(a|s)
$$

#### 合并得到贝尔曼期望方程
$$
\begin{aligned}
v_\pi(s)
&=\sum_{a}\pi(a|s)\sum_{r} p(r|s,a)\,r
+\gamma \sum_{a}\pi(a|s)\sum_{s'} p(s'|s,a)\,v_\pi(s') \\
&=\sum_{a}\pi(a|s)\left[
\sum_{r} p(r\mid s,a)\,r
+\gamma \sum_{s'} p(s'\mid s,a)\,v_\pi(s')
\right]
\end{aligned}
$$

|符号|含义|
|---|---|
|Gt|时刻t开始的总回报|
|$\gamma$|折扣系数 $0\le\gamma\le1$|
|$v_\pi(s)$|策略$\pi$下状态$s$的状态价值|

### 关键总结:MDP 与贝尔曼公式

- **MDP = 规则**:状态、动作、奖励、转移概率 $p(s'|s,a)$、策略 $\pi(a|s)$,描述"从哪到哪、得多少分"。
- **贝尔曼公式 = 按规则算每个状态值多少钱**:把无穷远的奖励拆成"这一步 + 下一个状态"的接力:

$$v(s) = r + \gamma \times v(s')$$

- **$\gamma$ = 眼光长短**:
 $\gamma \to 0$ 只看眼前
 $\gamma \to 1$ 看重长远。
- **马尔可夫性质不矛盾**:它抹掉的是**过去**,不是未来。未来仍很重要,只是"预测未来只需要当前状态、不需要历史"——所以当前状态才能代表整个未来,value 才有意义。

**一句话:** MDP 定义规则,贝尔曼公式靠"当前状态能代表未来"这个性质,把每个状态的价值算出来。


## 贝尔曼期望公式向量表示

### 贝尔曼期望方程：矩阵向量形式与策略评估
#### 1. 标量形式
策略 $\pi$ 下状态价值贝尔曼期望方程：

$$v_\pi(s) = \sum_a \pi(a|s)\left[
\sum_r p(r|s,a)\,r
+\gamma \sum_{s'} p(s'|s,a)\,v_\pi(s')
\right]$$

定义聚合期望：

$$\begin{align*}
r_\pi(s) &\triangleq \sum_a \pi(a|s)\sum_r p(r|s,a)\,r \\
P_\pi(s'|s) &\triangleq \sum_a \pi(a|s)\,p(s'|s,a)
\end{align*}$$
- $r_\pi(s)$：状态 $s$ 遵循策略 $\pi$ 的单步奖励期望
- $P_\pi(s'|s)$：策略诱导的状态转移概率

简化标量方程：

$$v_\pi(s) = r_\pi(s) + \gamma \sum_{s'} P_\pi(s'|s)\,v_\pi(s')$$

#### 2. 矩阵-向量形式
设一共有 $n$ 个状态 

$\{s_1,s_2,\dots,s_n\}$

价值向量、奖励向量：

$$\boldsymbol{v}_\pi =
\begin{bmatrix}
v_\pi(s_1)\\v_\pi(s_2)\\\vdots\\v_\pi(s_n)
\end{bmatrix},\quad
\boldsymbol{r}_\pi =
\begin{bmatrix}
r_\pi(s_1)\\r_\pi(s_2)\\\vdots\\r_\pi(s_n)
\end{bmatrix}$$

策略状态转移矩阵 
$\boldsymbol{P}_\pi\in\mathbb{R}^{n\times n}$

$$\big[\boldsymbol{P}_\pi\big]_{ij}=P_\pi(s_j \mid s_i)$$

紧凑矩阵贝尔曼方程：

$$\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma \boldsymbol{P}_\pi \boldsymbol{v}_\pi$$

### 闭式解析解

$$
\begin{aligned}
\boldsymbol{v}_\pi - \gamma \boldsymbol{P}_\pi \boldsymbol{v}_\pi &= \boldsymbol{r}_\pi \\
\left(\boldsymbol{I}-\gamma \boldsymbol{P}_\pi\right)\boldsymbol{v}_\pi &= \boldsymbol{r}_\pi \\
\boldsymbol{v}_\pi &= \left(\boldsymbol{I}-\gamma \boldsymbol{P}_\pi\right)^{-1}\boldsymbol{r}_\pi
\end{aligned}
$$

$\boldsymbol{I}$ 为单位矩阵。

#### 3. 迭代策略评估公式
迭代更新规则：

$$\boldsymbol{v}_{k+1} = \boldsymbol{r}_\pi + \gamma \boldsymbol{P}_\pi \boldsymbol{v}_k$$

收敛性质：

$$k\rightarrow\infty \implies \boldsymbol{v}_k \rightarrow \boldsymbol{v}_\pi$$

停止准则：

$$\max_{s}\left|\boldsymbol{v}_{k+1}(s)-\boldsymbol{v}_k(s)\right| < \theta$$

$\theta$ 为预设收敛阈值。

#### 4. 4状态MDP向量展开示例

$$\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}
=\begin{bmatrix}
r_\pi(s_1) \\
r_\pi(s_2) \\
r_\pi(s_3) \\
r_\pi(s_4)
\end{bmatrix}
+\gamma
\begin{bmatrix}
P_\pi(s_1|s_1) & P_\pi(s_2|s_1) & P_\pi(s_3|s_1) & P_\pi(s_4|s_1) \\
P_\pi(s_1|s_2) & P_\pi(s_2|s_2) & P_\pi(s_3|s_2) & P_\pi(s_4|s_2) \\
P_\pi(s_1|s_3) & P_\pi(s_2|s_3) & P_\pi(s_3|s_3) & P_\pi(s_4|s_3) \\
P_\pi(s_1|s_4) & P_\pi(s_2|s_4) & P_\pi(s_3|s_4) & P_\pi(s_4|s_4)
\end{bmatrix}
\begin{bmatrix}
v_\pi(s_1) \\
v_\pi(s_2) \\
v_\pi(s_3) \\
v_\pi(s_4)
\end{bmatrix}$$

#### 5. Action value(和state value区别就是有了动作action)

$$q_Π(s,a)=E[Gt|St=s,At=a]$$


$$E[G_t \mid S_t = s] = \sum_{a} E[G_t \mid S_t = s, A_t = a]\pi(a|s)$$

$$v_\pi(s) = \sum_{a} \pi(a|s)\,q_\pi(s,a)$$

### 疑惑答疑

## 1.为什么要学习贝尔曼公式：
用大白话讲，贝尔曼公式就是RL世界的标准答案模板，所有的强化学习都是按照这个标准答案不断优化

## 2.为什么既要学习state value还要学习action value 二者到底有什么用
state value 本质就是对"状态"的评估：站在s这个位置，按照当前策略，平均能拿多少分（未来reward的折扣期望）；Action value本质是对”状态+动作"的一个评估，站在s这个位置，选择a这个动作，平均能拿多少分，以此来找最优策略

## 3.贝尔曼公式在生活中有哪些具体的应用，怎么实现的
经过我的查阅，最典型的就是手机地图的导航，app每天都在算：从当前位置走哪条路到达终点是最快的。

- 状态=每个路口（节点）
- 动作=能走哪条路
- 奖励=走这条路需要花费时间（时间越短越好）
- 路口价值：从这个路口到终点，最快要多久


## 贝尔曼最优公式

### optimal policy

a policy Π* is optimal if v<sub>Π*</sub>(s) >=v<sub>Π</sub>(s)  for all s and for any other policy Π


## Bellman最优方程
标量形式

```math
v_{\ast}(s) = \max_{\pi}\sum_{a}\pi(a|s)\left(
\sum_{r}p(r|s,a)r
+\gamma\sum_{s'}p(s'|s,a)v_{\ast}(s')
\right)
```

```math
v_{\ast}(s) = \max_{\pi}\sum_{a}\pi(a|s)\;q_{\ast}(s,a)
```


矩阵向量形式

$$\boldsymbol{v}_{*} = \max_{\pi}\big(\boldsymbol{r}_{\pi} + \gamma P_{\pi}\boldsymbol{v}_{*}\big)$$

$$\begin{aligned}
\big[\boldsymbol{r}_{\pi}\big]_{s} &\triangleq \sum_{a}\pi(a|s)\sum_{r}p(r|s,a)r \\
\big[P_{\pi}\big]_{s,s'} &\triangleq \sum_{a}\pi(a|s)p(s'|s,a)
\end{aligned}$$

### 符号速记
- $v_*(s)$：最优状态价值函数
- $\max\limits_{\pi}$：对所有可行策略取最大化
- $\pi(a|s)$：策略
- $\gamma$：折扣因子
- $P_{\pi}$：策略诱导的状态转移矩阵
- $\boldsymbol{v}_*$：最优状态价值向量

### 贝尔曼最优公式的性质
- 不动点和压缩映射证明最优解和值迭代
#### 不动点
如果一个数x代入函数f之后，算出来结果还是x，也就是 $f(x)=x$，那这个x就是不动点。

#### 压缩映射
存在一个0到1之间的常数γ，对任意两个自变量$x_1,x_2$满足：
$|f(x_1)-f(x_2)| \le \gamma |x_1-x_2|$
1. 式子里面绝对值本质就是一种范数，范数用来衡量两者差距；
2. 含义：经过函数变换后，两点之间的差距被缩小；
3. 一维函数可以用绝对值来理解这个不等式。

#### 压缩映射定理
满足压缩映射条件的函数，存在唯一不动点；
随便选一个初始值不断迭代 $x_{k+1}=f(x_k)$，最终会收敛到这个不动点。


- factors determine the optimal policy:

    Reward design:r     
    System model:p    
    Discount rate:Γ

---

下一节:[02 反向传播](02-反向传播.md) ｜ 返回:[目录](README.md)
