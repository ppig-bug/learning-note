import random
from collections import deque
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn


# ===== 1. 经验回放（记忆库）=====
class ReplayBuffer:
    """存经历 (状态, 动作, 奖励, 下一状态, 是否结束)，训练时随机抽一小批"""
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)  # 用双端队列存经历；存满后自动丢掉最旧的

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))  # 打包成一个元组存进去

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)  # 不放回随机抽样
        s, a, r, s_next, done = zip(*batch)  # 把每条经历竖着拆开，按字段归成 5 组
        return (
            torch.tensor(np.stack(s), dtype=torch.float32),    # 先拼成数组再转张量（更快）
            torch.tensor(np.array(a), dtype=torch.long),
            torch.tensor(np.array(r), dtype=torch.float32),
            torch.tensor(np.stack(s_next), dtype=torch.float32),
            torch.tensor(np.array(done), dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ===== 2. Q 网络（神经网络代替 Q 表）=====
class QNetwork(nn.Module):
    """输入状态向量，输出每个动作的 Q 值"""
    def __init__(self, n_states: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),  # 输出层：每个动作一个 Q 值
        )

    def forward(self, x):
        return self.net(x)


# ===== 3. DQN Agent =====
class DQNAgent:
    def __init__(self, env, learning_rate: float = 1e-3, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, batch_size: int = 32,
                 buffer_size: int = 10000, target_update_freq: int = 100):
        self.env = env
        self.n_states = env.observation_space.shape[0]  # 状态维度 = 4（CartPole 是 4 维向量）
        self.n_actions = env.action_space.n
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # 两个网络：主网络（学）+ 目标网络（算 TD 目标）
        self.q_net = QNetwork(self.n_states, self.n_actions)
        self.target_net = QNetwork(self.n_states, self.n_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())  # 开局先复制主网络参数
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.num_timesteps = 0

    def predict(self, obs, deterministic: bool = False):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (4,) → (1, 4)
        with torch.no_grad():
            q_values = self.q_net(obs)
        if deterministic:
            return int(q_values.argmax().item())
        if np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(q_values.argmax().item())

    def learn(self, total_timesteps: int):
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_count = 0

        for step in range(total_timesteps):
            # 1. 选动作、执行、拿反馈
            action = self.predict(obs)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated  # 本局是否结束（布尔值）

            # 2. 存经验
            self.replay_buffer.push(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_reward += reward
            self.num_timesteps += 1

            # 3. 记忆库攒够了，随机抽一批学习
            if len(self.replay_buffer) >= self.batch_size:
                # 注意：批量里的 done 改名叫 done_mask，别覆盖上面的"本局结束"标记
                s, a, r, s_next, done_mask = self.replay_buffer.sample(self.batch_size)

                current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)  # Q(s,a)

                # TD 目标（只在算目标时关梯度，训练部分必须开梯度）
                with torch.no_grad():
                    next_q = self.target_net(s_next).max(dim=1)[0]
                    td_target = r + self.gamma * next_q * (1 - done_mask)  # 贝尔曼公式

                # 4. 损失 + 梯度下降（必须在 no_grad 外面）
                loss = self.loss_fn(current_q, td_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 5. 定期同步目标网络
                if self.num_timesteps % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

           
            if done:
                episode_count += 1
                if episode_count % 20 == 0:
                    print(f"Ep {episode_count} | Reward: {episode_reward:.1f} | "
                          f"ε: {self.epsilon:.3f} | 记忆库: {len(self.replay_buffer)}")
                obs, info = self.env.reset()
                episode_reward = 0.0
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return self


# ===== 4. 主程序：训练 + 评估 =====
if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = DQNAgent(env)
    agent.learn(total_timesteps=30000)

    eval_env = gym.make("CartPole-v1")
    rewards = []
    for _ in range(10):
        obs, info = eval_env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            total += reward
        rewards.append(total)
    print(f"\n评估：10 局平均奖励 = {np.mean(rewards):.1f}（CartPole 满分为 500）")
