# PPO 论文 Figure 2 复现：Hopper 上的一次策略更新
# 论文图里是四条曲线：
#   1) E_t[KL_t]
#   2) L^CPI = E_t[r_t * A_t]
#   3) E_t[clip(r_t, 1-eps, 1+eps) * A_t]
#   4) L^CLIP = E_t[min(r_t*A_t, clip(r_t, 1-eps, 1+eps)*A_t)]
#
# 横轴 t 是参数线性插值因子：
#   theta_t = (1-t) * theta_old + t * theta_new
# t=0 是更新前策略，t=1 是做完一次 PPO 更新后的策略。
# t 不是学习率，也不是训练步数，只用于把旧策略到新策略的这条直线扫一遍。

import copy
import math
import os

import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- 超参数 ----------------
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
N_STEPS = 2048
N_EPOCHS = 10
MINIBATCH = 64
LR = 3e-4
SEED = 0
DEVICE = torch.device("cpu")


def mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        for seq, gain in ((self.actor, 0.01), (self.critic, 1.0)):
            for layer in seq[:-1]:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                    nn.init.constant_(layer.bias, 0.0)
            nn.init.orthogonal_(seq[-1].weight, gain=gain)
            nn.init.constant_(seq[-1].bias, 0.0)

    def dist(self, obs):
        mean = self.actor(obs)
        std = self.log_std.clamp(-20.0, 2.0).exp().expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        dist = self.dist(obs)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp, self.value(obs)

    def evaluate(self, obs, action):
        dist = self.dist(obs)
        logp = dist.log_prob(action).sum(-1)
        return self.value(obs), logp, dist.entropy().sum(-1)


def collect(env, model, n_steps):
    obs_buf, act_buf, logp_buf = [], [], []
    rew_buf, val_buf, nextval_buf, cont_buf = [], [], [], []

    obs, _ = env.reset(seed=SEED)
    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=DEVICE)
        dist = model.dist(obs_t)
        action = dist.sample()
        action_np = np.clip(action.cpu().numpy(), env.action_space.low, env.action_space.high)

        # 环境实际执行的是裁剪后的动作，logp 也必须按同一个动作算；
        # 否则 t=0 时 ratio 都不等于 1，整张图会整体偏掉。
        action_used = torch.as_tensor(action_np, dtype=torch.float32, device=DEVICE)
        logp = dist.log_prob(action_used).sum(-1)
        value = model.value(obs_t)

        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        next_value = model.value(torch.as_tensor(next_obs, dtype=torch.float32, device=DEVICE))

        obs_buf.append(np.asarray(obs, dtype=np.float32))
        act_buf.append(np.asarray(action_np, dtype=np.float32))
        logp_buf.append(float(logp.item()))
        rew_buf.append(float(reward))
        val_buf.append(float(value.item()))
        nextval_buf.append(0.0 if terminated else float(next_value.item()))
        cont_buf.append(0.0 if terminated else 1.0)

        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    return (
        np.asarray(obs_buf, dtype=np.float32),
        np.asarray(act_buf, dtype=np.float32),
        np.asarray(logp_buf, dtype=np.float32),
        rew_buf,
        np.asarray(val_buf, dtype=np.float32),
        np.asarray(nextval_buf, dtype=np.float32),
        np.asarray(cont_buf, dtype=np.float32),
    )


def gae(rewards, values, next_values, cont):
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        delta = rewards[t] + GAMMA * next_values[t] * cont[t] - values[t]
        running = delta + GAMMA * LAM * cont[t] * running
        advantages[t] = running
    return advantages


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    env = gym.make("Hopper-v4")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = ActorCritic(obs_dim, act_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 1. 用 theta_old 采样一批数据
    obs_np, act_np, logp_np, rewards, values, next_values, cont = collect(env, model, N_STEPS)
    advantages = gae(rewards, values, next_values, cont)
    returns = advantages + values

    obs = torch.as_tensor(obs_np, dtype=torch.float32, device=DEVICE)
    actions = torch.as_tensor(act_np, dtype=torch.float32, device=DEVICE)
    logp_old = torch.as_tensor(logp_np, dtype=torch.float32, device=DEVICE)

    # 优势全批标准化一次；训练和画图都用同一份，避免两边口径不一致
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

    # 2. 保存 theta_old，然后用 PPO 的 clip 目标做一次完整更新
    theta_old = copy.deepcopy(model.state_dict())

    for _ in range(N_EPOCHS):
        permutation = torch.randperm(N_STEPS)
        for start in range(0, N_STEPS, MINIBATCH):
            index = permutation[start:start + MINIBATCH]

            value_pred, logp, entropy = model.evaluate(obs[index], actions[index])
            ratio = torch.exp(logp - logp_old[index])

            surrogate_unclipped = ratio * advantages_t[index]
            surrogate_clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantages_t[index]
            policy_loss = -torch.min(surrogate_unclipped, surrogate_clipped).mean()

            value_loss = F.mse_loss(value_pred, returns_t[index])
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    theta_new = copy.deepcopy(model.state_dict())

    # 3. 记录 theta_old 在每个状态上的动作分布，后面计算精确 KL
    with torch.no_grad():
        model.load_state_dict(theta_old)
        old_dist = model.dist(obs)
        old_mean = old_dist.mean.clone()
        old_std = old_dist.stddev.clone()

    def interpolate(t):
        return {key: (1.0 - t) * theta_old[key] + t * theta_new[key] for key in theta_old}

    ts = np.linspace(0.0, 1.0, 41)
    kl_values = []
    cpi_values = []
    clipped_term_values = []
    clip_values = []

    # 4. 沿着 theta_old -> theta_new 扫 41 个点，分别算四条曲线
    with torch.no_grad():
        for t in ts:
            model.load_state_dict(interpolate(float(t)))

            dist = model.dist(obs)
            logp = dist.log_prob(actions).sum(-1)
            ratio = torch.exp(logp - logp_old)

            unclipped = ratio * advantages_t
            clipped = torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP) * advantages_t

            cpi_values.append(unclipped.mean().item())
            clipped_term_values.append(clipped.mean().item())
            clip_values.append(torch.min(unclipped, clipped).mean().item())

            # KL( N(mu_old, std_old) || N(mu_t, std_t) )，再对所有状态维度、状态样本求平均
            kl_per_dim = (
                torch.log(old_std / dist.stddev)
                + (dist.stddev.square() + (dist.mean - old_mean).square())
                / (2.0 * old_std.square())
                - 0.5
            )
            kl_values.append(kl_per_dim.sum(-1).mean().item())

    # 5. 画论文 Figure 2 的四条曲线
    os.makedirs("results", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(ts, kl_values, lw=2.4, color="#2ca02c", label=r"$\hat{E}_t[KL_t]$")
    ax.plot(ts, cpi_values, lw=2.4, color="#1f77b4",
            label=r"$L^{CPI}=\hat{E}_t[r_t A_t]$")
    ax.plot(ts, clipped_term_values, "--", lw=2.1, color="#ff7f0e",
            label=r"$\hat{E}_t[\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t]$")
    ax.plot(ts, clip_values, ":", lw=2.9, color="#d62728",
            label=r"$L^{CLIP}=\hat{E}_t[\min(r_t A_t,\mathrm{clip}(r_t)A_t)]$")

    ax.set_xlabel("Linear interpolation factor")
    ax.set_ylabel("value")
    ax.set_title("PPO Figure 2 reproduction (Hopper-v4)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("results/fig2_four_lines.png", dpi=150, bbox_inches="tight")

    print("t = 1 values")
    print("E_t[KL_t]     =", kl_values[-1])
    print("L^CPI         =", cpi_values[-1])
    print("E[clip(r)A]   =", clipped_term_values[-1])
    print("L^CLIP        =", clip_values[-1])
    print("saved: results/fig2_four_lines.png")


if __name__ == "__main__":
    main()

