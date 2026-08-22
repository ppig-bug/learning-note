import gymnasium as gym 
from environment import GridWorldEnv
from qlearningagent import QLearningAgent

# env = GridWorldEnv()
env = gym.make("FrozenLake-v1", is_slippery=False)
model = QLearningAgent(env)
model.learn(total_timesteps=30000)  # 训练步数，可按需调整；原来 3000000 步会跑很久
