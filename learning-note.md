# 强化学习的数学原理

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
