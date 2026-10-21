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
$$
