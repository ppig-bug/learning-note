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