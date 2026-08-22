import numpy as np
from typing import Optional,Union,Tuple #给函数参数标注类型
import gymnasium as gym
import random
import pickle #把模型序列化保存成文件

class QLearningAgent:
    """
    遵循SB3(RL的工具库)核心契约：
        -learn(total_timestep=10000) 主训练入口（按时间步）
        -predict(observation) 推理入口
        -save(path) / load 保存/加载持久化
    """

    def __init__(self,env:gym.Env,learning_rate:float=0.1,
                 gamma:float=0.95,
                 epsilon_decay:float=0.9997,
                 exploration_initial_eps:float=1.0,
                 exploration_final_eps:float=0.05,
                 verbose: int=1,#日志详细程度，0静默，1正常打印 
                 ):
        self.env=env
        self.verbose=verbose
        self.num_timesteps=0
        self.n_states=env.observation_space.n
        self.n_actions=env.action_space.n
        self.learning_rate=learning_rate
        self.gamma=gamma
        self.epsilon_decay=epsilon_decay
        self.exploration_initial_eps=exploration_initial_eps
        self.exploration_final_eps=exploration_final_eps
        self.q_table=np.zeros((self.n_states,self.n_actions),dtype=np.float64)
        self._current_eps=exploration_initial_eps

    def predict(
            self,
            observation:Union[int,np.ndarray],
            state:Optional[Tuple[np.ndarray,...]]=None,
            episode_start:Optional[np.ndarray]=None,
            deterministic:bool=False,
    ) -> Tuple[np.ndarray,Optional[Tuple[np.ndarray,...]]]:
        """
        deterministic 参数用于控制动作选择是否包含随机性
        True表示确定性动作，可复现，False表示可以探索动作用于训练

        """
        #obs就是Q值提到的state
        #目的：观测格式归一化，环境有时候返回数组，有时返回整数，统一转换成obs
        if isinstance(observation,np.ndarray):
            obs=int(observation.item()) if observation.ndim ==0 else int(observation[0])
        else:
            obs=int(observation)

        if deterministic:
            #当前状态下选择使得Q值最大的动作
            #Q-learning算法中ε-greedy 贪心选取最优动作
            q_values=self.q_table[obs]#取出当前状态obs所有动作对应的Q值一维数组
            max_q=q_values.max()
            candidates=np.where(q_values==max_q)[0]#找出所有等于最大值的下标
            action=int(np.random.choice(candidates))#在并列的最优里面随机挑选一个
        else:
            # ε-greedy 策略：探索 vs 利用
            random_num=np.random.rand()
            if random_num<self._current_eps:
                action=random.randrange(self.n_actions)
            else:
                q_values=self.q_table[obs]
                max_q=q_values.max()
                candidates=np.where(q_values==max_q)[0]
                action=int(np.random.choice(candidates))
        return np.array([action]),state

    def learn(self,total_timesteps:int,
              callback=None,
              log_interval:int=100,):
        obs,info=self.env.reset()
        episode_reward=0.0
        episode_count=0#局数计数器

        for step in range(total_timesteps):
            action_arr,_=self.predict(obs,deterministic=False)
            action=int(action_arr[0])
            next_obs,reward,terminated,truncated,info=self.env.step(action)
            done=terminated or truncated

            #核心更新公式
            current_q=self.q_table[obs,action]
            best_next_q=self.q_table[next_obs].max() if not terminated else 0
            td_target=reward+self.gamma*best_next_q
            td_error=td_target-current_q
            self.q_table[obs,action]+=self.learning_rate*td_error

            episode_reward+=reward
            self.num_timesteps+=1

            #Episode结束处理
            if done:
                episode_count+=1
                if self.verbose>=1 and episode_count%log_interval==0:
                    print(f"Timestep{self.num_timesteps}/{total_timesteps} | "
                        f"Ep {episode_count} | Reward: {episode_reward:.2f} | "
                          f"ε: {self._current_eps:.4f}")

                #重置
                obs,info=self.env.reset()
                episode_reward=0.0
                self._current_eps=max(self.exploration_final_eps,self._current_eps*self.epsilon_decay)
            else:
                obs=next_obs #没结束就带着新观测进入下一步循环
        return self


    def save(self,path:str) ->None:
        #保存模型SB3使用.zip,这里简化为pickle
        data={
            "q_table": self.q_table,
            "num_timesteps": self.num_timesteps,
            "_current_eps": self._current_eps,
            "config": {
                "learning_rate": self.learning_rate,
                "gamma": self.gamma,
                "epsilon_decay": self.epsilon_decay,
                "exploration_initial_eps": self.exploration_initial_eps,
                "exploration_final_eps": self.exploration_final_eps,
            },
        }

        with open(path,'wb')as f:
            pickle.dump(data,f)

    @classmethod#表示由类本身调用
    def load(cls, path: str, env=None) :
        #加载模型（SB3 惯例: load 是 classmethod，env 可选传入）
        with open(path, "rb") as f:
            data = pickle.load(f)

        config = data["config"]#取出超参数字典
        agent = cls(env=env, **config)#用超参数新建一个全新的agent
        #新建再填原因：第三步会走一遍init,而init只会给你一张全新为0的Q表
        agent.q_table = data["q_table"]
        agent.num_timesteps = data["num_timesteps"]
        agent._current_eps = data["_current_eps"]
        return agent

    # ══════════════════════════════════════════════
    #  工具方法
    # ══════════════════════════════════════════════

    def get_q(self, state: int, action: int) -> float:
        return float(self.q_table[state, action])

    def __repr__(self) -> str:
        return (
            f"TabularQLearning(states={self.n_states}, actions={self.n_actions}, "
            f"timesteps={self.num_timesteps}, ε={self._current_eps:.4f})"
        )


if __name__ == '__main__':
    from environment import GridWorldEnv
    env = GridWorldEnv()
    agent = QLearningAgent(env)

    print(agent)                        # 打印智能体摘要（用了 __repr__）
    agent.learn(total_timesteps=2000)   # 训练 2000 步，会打印训练日志
    agent.save("agent.pkl")             # 保存模型

    # 加载并测试
    agent2 = QLearningAgent.load("agent.pkl", env=env)
    obs, info = env.reset()
    action, _ = agent2.predict(obs, deterministic=True)
    print(f"初始状态: {obs}, 选择动作: {action}")
