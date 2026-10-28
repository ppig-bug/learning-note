## REINFORCE 策略梯度推导

% 1. 目标期望

$$J(\theta)=\mathbb{E}_{\tau\sim P_\theta(\tau)}[R(\tau)]=\sum_{\tau} R(\tau)P_\theta(\tau)$$

% 2. 求梯度

$$\begin{aligned}
\nabla J(\theta)&=\nabla\sum_{\tau} R(\tau)P_\theta(\tau)\\
&=\sum_{\tau} R(\tau)\,\nabla P_\theta(\tau)
\end{aligned}$$

% 3. 恒等变形，同乘同除概率

$$\begin{aligned}
\nabla J(\theta)&=\sum_{\tau} R(\tau)\nabla P_\theta(\tau)\cdot\frac{P_\theta(\tau)}{P_\theta(\tau)}\\
&=\sum_{\tau} P_\theta(\tau)\,R(\tau)\,\frac{\nabla P_\theta(\tau)}{P_\theta(\tau)}
\end{aligned}$$

% 对数梯度恒等式

$$\nabla\ln P_\theta(\tau)=\frac{\nabla P_\theta(\tau)}{P_\theta(\tau)}$$

% 定义随机变量

$$X(\tau)=R(\tau)\frac{\nabla P_\theta(\tau)}{P_\theta(\tau)}$$

$$\nabla J(\theta)=\mathbb{E}_{\tau\sim P_\theta}\big[X(\tau)\big]=\sum_{\tau} P_\theta(\tau)X(\tau)$$

% 4. 蒙特卡洛样本近似

$$\mathbb{E}\big[X(\tau)\big]\approx\frac1N\sum_{n=1}^{N}X(\tau^n)$$

$$\begin{aligned}
\nabla J(\theta)&\approx\frac1N\sum_{n=1}^{N} R(\tau^n)\,\frac{\nabla P_\theta(\tau^n)}{P_\theta(\tau^n)}\\
&\approx\frac1N\sum_{n=1}^{N} R(\tau^n)\,\nabla\ln P_\theta(\tau^n)
\end{aligned}$$

% 5. 轨迹概率因式分解

$$P_\theta(\tau^n)=\prod_{t=1}^{T_n} P_\theta(a_t^n\mid s_t^n)$$

$$\ln P_\theta(\tau^n)=\sum_{t=1}^{T_n}\log P_\theta(a_t^n\mid s_t^n)$$

$$\nabla\ln P_\theta(\tau^n)=\sum_{t=1}^{T_n}\nabla\log P_\theta(a_t^n\mid s_t^n)$$

% 6. REINFORCE最终梯度
$$\nabla J(\theta)\approx\frac1N\sum_{n=1}^{N} R(\tau^n)\sum_{t=1}^{T_n}\nabla\log P_\theta(a_t^n\mid s_t^n)$$


> 通过公式可以看出有改进的空间；1.能否增大在状态s下做动作a的概率不应该只看当前，而是一个全局累积的过程，也就是说一个动作只能影响后面的，影响不了前面的；2.一个reward可能只对周围（后面）的reward产生影响，越到后面可能影响越微弱

$$\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}\big(R_t^n - B(s_t^n)\big)\nabla\log P_\theta(a_n^t|s_n^t)$$

原始策略梯度，$R_t^n-B(s_t^n)$ 是**带基线的回报**，基线$B$用来降低梯度估计方差。


**Action-Value Function**

$R_t^n$ 每次都是一次随机采样，方差很大，训练不稳定。

$Q_\theta(s,a)$ 在state $s$ 下，做出Action $a$，期望的回报。动作价值函数。

**State-Value Function**
$V_\theta(s)$ 在state $s$ 下，期望的回报。状态价值函数。

**Advantage Function**

$A_\theta(s,a) = Q_\theta(s,a) - V_\theta(s)$ 在state $s$ 下，做出Action $a$，比其他动作能带来多少优势。

$$\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n}A_\theta(s_n^t,a_n^t)\nabla\log P_\theta(a_n^t|s_n^t)$$

## 邻近策略优化PPO(on policy)


### 重要性采样

假设我们有一个函数f（x），假设不能从分布p采样数据，只能从另外一个分布q采样数据x,q可以是任意分布，期望值Ex∼p[f(x)]就是∫f(x)p(x)dx

$$\int f(x)p(x)\mathrm{d}x = \int f(x)\frac{p(x)}{q(x)}q(x)\mathrm{d}x = \mathbb{E}_{x\sim q}\left[f(x)\frac{p(x)}{q(x)}\right]$$

就可得

$$\mathbb{E}_{x\sim p}\left[f(x)\right] = \mathbb{E}_{x\sim q}\left[ f(x)\frac{p(x)}{q(x)} \right] \tag{5.3}$$

就可以写成对q里面所采样出来的x取期望值，这样就解决了即使我们不能从p里面采样数据，但只要从q里面采样数据，就可以算出p采样后的期望值


### 重要性权重ratio

> 就是用旧策略的数据评估新策略时，给每个样本打的“折算系数”：

$$(w=\frac{\pi_{\text{new}}(a\mid s)}{\pi_{\text{old}}(a\mid s)})$$

1. 重要性采样修正什么偏差？


策略梯度的原始目标：

$$(\nabla J=\mathbb{E}_{a\sim\pi_\theta}\left[\nabla\log\pi_\theta(a\mid s)\,A(s,a)\right])$$

但是这个期望的下标是必须从 $\pi_\theta$采样，只有旧数据最后只能算出旧策略的梯度

解决方法：

$$(\mathbb{E}_{a\sim\pi_{\text{new}}}[f(a)]=\sum_a \pi_{\text{new}}(a)f(a)=\sum_a \pi_{\text{old}}(a)\underbrace{\frac{\pi_{\text{new}}(a)}{\pi_{\text{old}}(a)}}_{w}f(a)=\mathbb{E}_{a\sim\pi_{\text{old}}}[w\cdot f(a)])$$

2. 期望没问题，但是方差会爆炸:

$p$ 和 $q$ 差得越多，重要性采样效果越差

例如：
$p=\mathcal N(\mu,1)$，$q=\mathcal N(0,1)$，估计 $\mathbb E_p[x]$，真值就是 $\mu$，采 1000 个样本）：

$$(\mathbb E_q[w]=1,\qquad \mathbb E_q[w^2]=e^{\mu^2})$$

均值永远是 1（无偏），但二阶矩指数增长。 方差 = 二阶矩 − 均值²，所以方差也是指数级。

3. PPO的循环：
$$
[
\text{采数据} \rightarrow \text{用旧数据把 } \pi \text{ 推几步} \rightarrow \pi \text{ 变了} \rightarrow \text{旧数据过期} \rightarrow \text{重新采}
]

### PPO的前身：trust region policy optimization(TRPO)

1. TRPO与PPO不一样的地方：约束位置，PPO直接把约束放在最优化的式子里面，而TRPO是把KL散度当成约束，希望θ 与θ'的KL散度小于δ

2. TRPO难处理的原因：把KL散度约束当成一个额外的约束，没有放在目标里面，很难计算

3. KL散度：输入θ与θ'，计算这两个之间的行为距离。**行为距离**指的是给定同样的状态，输出动作之间的差距




```
  强化学习 PPO 复习思路整理

1. 基础定义：回报、价值、优势函数
累积回报
$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots$$

价值函数
- $V(s) = \mathbb{E}_\pi[G_t \mid s_t=s]$：**状态价值**，处于状态s能获得的期望总回报
- $Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid s_t=s,a_t=a]$：**动作价值**，状态s执行动作a后的期望总回报

优势函数
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$
> $A>0$：这个动作比平均更好，提高动作概率
> $A<0$：这个动作更差，降低动作概率

$$Q^\pi(s,a) = \mathbb{E}_{s'\sim P(\cdot|s,a)}\left[ r_t + \gamma \cdot V^\pi(s') \right]$$
> 当前即时奖励 + 下一状态折扣价值

TD 时序差分误差
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$
> Critic网络预测值，和真实估计回报之间的差值

2. Actor-Critic 双网络结构
- Actor（策略网络 $\pi_\theta(a|s)$）：输出动作分布，负责选动作
- Critic（价值网络 $V_\phi(s)$）：评估当前状态好不好

损失函数
- Critic损失（拟合状态价值）
$$L_{critic} = \big(V_\phi(s_t) - \hat{G}_t\big)^2$$

- Actor原始损失（策略梯度）
$$\max \mathbb{E}_{s,a\sim\pi_\theta}\big[\log\pi_\theta(a|s)\cdot \hat A_t\big]$$

GAE 广义优势估计（PPO标配）
$$\hat A_t^{GAE(\gamma,\lambda)} = \sum_{i=0}^{\infty} (\gamma\lambda)^i \delta_{t+i}$$
$$\hat G_t = \delta_t + \gamma\lambda \hat G_{t+1}$$
- $\gamma$：折扣因子
- $\lambda$：GAE系数，用来权衡偏差与方差

> 流程：收集一批轨迹数据 → 计算GAE优势 → 多轮更新网络

3. 重要性采样
旧策略收集轨迹，训练新策略
$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$
- $r_t>1$：新策略选择该动作概率高于旧策略

4. TRPO 信任区域策略优化
目标：
$$\max_\theta \mathbb{E}\big[ r_t(\theta) \hat A_t \big]$$
约束：
$$\mathbb{E}\big[ D_{KL}\big(\pi_{old}(\cdot|s) \ \|\ \pi_\theta(\cdot|s)\big) \big] \le \varepsilon$$
> KL散度约束：新旧策略差距不能太大，防止训练崩掉

两种实现：
1. TRPO-Penalty 惩罚版本
$$L_{KLPEN}(\theta) = \mathbb{E}\big[ r_t(\theta)\hat A_t - \beta \cdot D_{KL}(\pi_{old} \Vert \pi_\theta) \big]$$
> 自适应调整$\beta$，KL散度变大就增强约束

2. TRPO-Clip：裁剪思想，是PPO的前身

5. PPO 近端策略优化
PPO-CLIP（最常用）
$$L^{CLIP}(\theta) = \mathbb{E}\Big[ \min\big( r_t(\theta)\hat A_t,\ \text{clip}(r_t(\theta), 1-\varepsilon,1+\varepsilon)\cdot \hat A_t \big) \Big]$$
> clip：将比率$r_t$限制在$[1-\varepsilon,\ 1+\varepsilon]$，限制策略更新步长

PPO 完整总损失（Actor + Critic + 熵正则）
$$L^{PPO}(\theta) = \mathbb{E}\Big[ L_t^{CLIP}(\theta) - c_1 \big(V_\phi(s_t)-\hat G_t\big)^2 + c_2 H\big[\pi_\theta(\cdot|s_t)\big] \Big]$$
- $L_t^{CLIP}$：Actor裁剪目标函数
- $(V_\phi(s_t)-\hat G_t)^2$：Critic价值损失
- $H$ 熵正则项：鼓励策略保持探索，避免过早收敛到确定性动作
```
