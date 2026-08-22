import numpy as np
import gymnasium as gym #标准的强化学习环境库，负责提供FrozenLake

#1.初始化
#FrozenLake-v1环境，4*4冰湖，起点在左上角，重点在右下角，中间有陷阱洞
env=gym.make("FrozenLake-v1",is_slippery=False)#表示冰面不滑，动作是确定的，按那个方向走就走到哪，改成True会变成随机环境
n_states=env.observation_space.n #16个格子
n_actions=env.action_space.n #4个方向

print(n_states)
print(n_actions)

#Q表：16行4列，初始全为0
Q_tables=np.zeros((n_states,n_actions))

#超参数
alpha=0.1
gamma=0.95 
epsilon=1.0#初始探索率
epsilon_decay=0.9997#每局训练后探索率的衰减系数
epsilon_min=0.1#探索率的下限
episodes=30000#一共训练30000局
rewards_history=[]

#2.训练循环
for episode in range(episodes):
    state,info=env.reset()#每局开始，把环境重置到初始状态
    total_reward=0
    done=False#标记本局是否结束

    while not done:
        random_num=np.random.rand()
        if random_num<epsilon:#探索
            action=env.action_space.sample() #在四个动作里面随机挑选一个
        else:#利用
            action=np.argmax(Q_tables[state])

        next_state,reward,terminated,truncated,info=env.step(action)#把动作交给环境执行，返回5个值
        done=terminated or truncated

        #贝尔曼公式
        best_next_q=np.max(Q_tables[next_state]) if not terminated else 0
        td_target=reward+gamma*best_next_q
        td_error=td_target-Q_tables[state,action]
        Q_tables[state,action]+=alpha*td_error

        #迭代
        state=next_state
        total_reward+=reward

        #衰减探索率
        epsilon=max(epsilon_min,epsilon*epsilon_decay)
        rewards_history.append(total_reward)

        if(epsilon+1)%200==0:
            avg=np.mean(reward_history[-100:])#最近100局的平均奖励
            print(f"Episode {episode+1} | Avg Reward: {avg:.3f} | Epsilon:{epsilon:.3f}")

#3.查看学到的Q表
for s in range(n_states):
    print(f"State{s:2d}:{np.round(Q_tables[s],3)}")