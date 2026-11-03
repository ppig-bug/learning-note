import gymnasium as gym
import torch.nn as nn
import torch
import math
import csv
import torch.nn.functional as F
from torch.distributions import Categorical,Normal#离散动作以及连续高斯
import os
import numpy as np
import argparse#命令行参数解释器
"""
环境obs
Actor输出动作分布
采样action，记录logp_old
Critic输出V(s)
env.step()返回reward,next_obs,terminated,truncated
GAE算优势
同一批数据多轮复用
算重要性权重ratio=exp(new_logp-lohp_old)
critic更新Actor
"""

def mlp(in_dim,out_dim,hidden=64):
    return nn.Sequential(
            nn.Linear(in_dim,hidden),
            nn.Tanh(),#在连续控制里面更稳
            nn.Linear(hidden,hidden),
            nn.Tanh(),
            nn.Linear(hidden,out_dim),
    )


class ActorCritic(nn.Module):
    #Actor 离散输出logits，连续输出高斯分布的均值mean
    #Critic输出价值状态V(s)
    def __init__(self,obs_dim,action_dim,continuous):
        super().__init__()
        self.continuous=continuous
        self.actor=mlp(obs_dim,action_dim)
        self.critic=mlp(obs_dim,1)
        if continuous:
            self.log_std=nn.Parameter(torch.zeros(action_dim))#计算log标准差，初始化为0时std=1
        self._init_weights()#自定义初始化权重

    def _init_weights(self):
        #隐藏层gain=sqrt(2);Actor的gain为0.01
        def init_hidden(seq):
            for layer in seq[:-1]:#遍历除了最后一层之外的所有层
                if isinstance(layer,nn.Linear):#线性层
                    nn.init.orthogonal_(layer.weight,gain=math.sqrt(2.0)) #初始正交化
                    nn.init.constant_(layer.bias,0.0) #偏置初始化设为0

        init_hidden(self.actor)
        init_hidden(self.critic)
        nn.init.orthogonal_(self.actor[-1].weight,gain=0.01)
        nn.init.constant_(self.actor[-1].bias,0.0)
        nn.init.orthogonal_(self.critic[-1].weight,gain=1.0)
        nn.init.constant_(self.critic[-1].bias,0.0)


    def forward(self,obs):
        return self.actor(obs),self.critic(obs).squeeze(-1)
     #前向传播返回动作和状态价值，squeeze把【batch，-1】压缩成【batch】
    
    def distribution(self,obs):
        actor_out,value=self.forward(obs)
        #先判断是连续动作还是离散动作，连续则输出高斯分布，离散输出logits
        if self.continuous:
            mean=actor_out
            std=self.log_std.clamp(-20,2).exp().expand_as(mean)#先压缩再转对数再和mean对齐
            return Normal(mean,std),value
        return Categorical(logits=actor_out),value

    def act(self,obs,deterministic=False):#采样动作
        dist,value=self.distribution(obs)
        if deterministic:#如果是评估
            if self.continuous:
                action=dist.mean
            else:
                action=torch.argmax(dist.logits,dim=-1)#沿着最后一维取argmax
        else:
            action=dist.sample()

        logp=dist.log_prob(action)
        if self.continuous:
            logp=logp.sum(-1)#如果是连续的，得到各位概率和
        return action,logp,value #返回动作，旧概率，还有价值

    def evaluate_actions(self,obs,actions): #训练时评估动作
        dist,value=self.distribution(obs)
        logp=dist.log_prob(actions)
        entropy=dist.entropy() #求当前策略的熵，用于探索鼓励
        if self.continuous:
            logp=logp.sum(-1)
            entropy=entropy.sum(-1)
        return value,logp,entropy

    @torch.no_grad()
    def predict_values(self,obs):#无动作预测价值
        return self.critic(obs).squeeze(-1)

def compute_GAE(rewards,values,next_values,cont,gamma,lam):#cont是每个时间步上面的0，1开关，终止了就是0，GAE不再往后估算
    #回顾式子
    #delta_t=r_t+gamma*V(s')-V(s_t)  A_t=delta_t+gamma*lam*A{t+1}
    T=len(rewards)
    adv=np.zeros(T,dtype=np.float32) #分配优势数组
    last_GAE=0.0#最后一个时间步没有优势

    for t in reversed(range(T)):#这里要注意必须倒序，因为r_t依赖r_t+1
        delta=rewards[t]+gamma*next_values[t]*cont[t]-values[t]
        last_GAE=delta+gamma*lam*cont[t]*last_GAE
        adv[t]=last_GAE

    return adv#返回整段优势

def env_action(action,continuous,action_space):#把torch动作转换成环境匹配的格式
    if continuous:
        a=action.detach().cpu().numpy().astype(np.float32)
        return np.clip(a,action_space.low,action_space.high)
    return int(action.item())

@torch.no_grad()
def evaluate(model,env,device,episodes=5):
    returns=[]
    for _ in range(episodes):
        obs,_ =env.reset()
        done=False
        ep_return=0.0 #本回合累计回报
        while not done:
            obs_t=torch.as_tensor(obs,dtype=torch.float32,device=device)#numpy观测转张量
            action,_,_=model.act(obs_t,deterministic=True)
            action_np=env_action(action,model.continuous,env.action_space)
            obs,reward,terminated,truncated,_=env.step(action_np)
            done=terminated or truncated
            ep_return+=float(reward)
        returns.append(ep_return)
    return float(np.mean(returns))


def main():                                                          
    p = argparse.ArgumentParser()                                    # 创建参数解析器
    p.add_argument("--env", type=str, default="CartPole-v1")         # 环境名，默认 CartPole
    p.add_argument("--total-steps", type=int, default=100_000)       # 总共交互多少个环境步
    p.add_argument("--n-steps", type=int, default=256)               # 每次采样多少步再更新
    p.add_argument("--n-epochs", type=int, default=10)               # 同一批数据训练几轮
    p.add_argument("--minibatch", type=int, default=64)              # 每个小批量多大
    p.add_argument("--lr", type=float, default=3e-4)                 # 学习率
    p.add_argument("--gamma", type=float, default=0.99)              # 折扣因子
    p.add_argument("--lam", type=float, default=0.95)                # GAE 的 lambda
    p.add_argument("--clip", type=float, default=0.2)                # PPO 裁剪范围 epsilon
    p.add_argument("--ent-coef", type=float, default=0.01)           # 熵奖励系数
    p.add_argument("--vf-coef", type=float, default=0.5)             # 价值损失系数
    p.add_argument("--max-grad-norm", type=float, default=0.5)       # 梯度裁剪阈值
    p.add_argument("--seed", type=int, default=0)                    # 随机种子
    p.add_argument("--eval-every-steps", type=int, default=20_480)   # 每多少步评估一次
    p.add_argument("--eval-episodes", type=int, default=5)           # 每次评估跑几个回合
    p.add_argument("--device", type=str, default="cpu")              # 计算设备，写 auto 自动选显卡
    p.add_argument("--save-csv", type=str, default="")               # 结果保存路径，空则不保存
    p.add_argument("--label", type=str, default="own")               # 写进 CSV 的算法名
    args = p.parse_args()                                            # 真正解析命令行

    if args.device == "auto":                                        # 用户要求自动选设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    else:                                                            # 用户指定了设备
        device = torch.device(args.device)                           # 直接按字符串构造设备对象

    torch.manual_seed(args.seed)                                     # 固定 torch 随机种子，保证可复现
    np.random.seed(args.seed)                                        # 固定 numpy 随机种子

    if args.save_csv:                                                
        csv_dir = os.path.dirname(os.path.abspath(args.save_csv))    
        os.makedirs(csv_dir, exist_ok=True)                          
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:  
            writer = csv.writer(f)                                   
            writer.writerow(["algo", "seed", "step", "eval_return"])  

    env=gym.make(args.env)
    eval_env=gym.make(args.env)
    obs_dim=int(np.prod(env.observation_space.shape))
    continuous=isinstance(env.action_space,gym.spaces.Box) #Box表示连续动作空间
    action_dim=int(env.action_space.shape[0]) if continuous else int(env.action_space.n)
    model=ActorCritic(obs_dim,action_dim,continuous).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    num_updates=max(1,args.total_steps//args.n_steps)#更新次数
    eval_every_updates=max(1,args.eval_every_steps//args.n_steps) #重新评估间隔
    obs,_=env.reset(seed=args.seed)
    best=-float("inf")#记录历史最好的评估成绩

    print(                                                           # 打印一行环境信息，方便确认配置
        f"env={args.env}  obs_dim={obs_dim}  action_dim={action_dim}  "  # 环境名、观测维度、动作维度
        f"continuous={continuous}  device={device}"                  # 是否连续动作、用的什么设备
    ) 

    for update in range(1,num_updates+1):
        obs_buf=[]
        act_buf=[]
        logp_buf=[]
        rew_buf=[]
        val_buf=[]
        nextval_buf=[]
        cont_buf=[]

        #1.采样
        for _ in range(args.n_steps):
            obs_t=torch.as_tensor(obs,dtype=torch.float32,device=device)
            with torch.no_grad():#采样不需要梯度
                action,logp,value=model.act(obs_t,deterministic=False)
            action_np=env_action(action,continuous,env.action_space)
            next_obs,reward,terminated,truncated,_=env.step(action_np)
            next_obs_t=torch.as_tensor(next_obs,dtype=torch.float32,device=device)
            next_value=model.predict_values(next_obs_t) #算GAE要用到的

            obs_buf.append(np.asarray(obs, dtype=np.float32).reshape(-1))  # 存观测，reshape 保证是 1 维
            act_buf.append(action.detach().cpu().numpy())            # 存动作，转 numpy
            logp_buf.append(float(logp.item()))                      # 存旧策略 log 概率，作为固定参考
            rew_buf.append(float(reward))                            # 存奖励
            val_buf.append(float(value.item()))  

            # 注意 terminated是真的结束，truncated没有结束只是超时间了
            nextval_buf.append(0.0 if terminated else float(next_value.item())) 
            cont_buf.append(0.0 if terminated else 1.0) 

            obs=next_obs
            if terminated or truncated:
                obs,_=env.reset()

            #2.算GAE
        adv=compute_GAE(
                rew_buf,
                val_buf,
                nextval_buf,
                cont_buf,
                args.gamma,
                args.lam,
            )
        returns=adv+np.asarray(val_buf,dtype=np.float32)#critic的目标就是优势+旧价值

        obs_batch = torch.as_tensor(                                # 观测批量转张量
            np.asarray(obs_buf), dtype=torch.float32, device=device  # numpy 数组转 float32 张量
        )                                                            
        act_batch = torch.as_tensor(                                # 动作批量转张量
            np.asarray(act_buf),                                     # 先把列表转成数组
            dtype=torch.float32 if continuous else torch.long,       # 连续用 float，离散用 long
            device=device,                                           
        )                                                            
        logp_old_batch = torch.as_tensor(                            # 旧 log 概率转张量
            np.asarray(logp_buf), dtype=torch.float32, device=device  # 统一成 float32
        )                                                            
        adv_batch = torch.as_tensor(adv, dtype=torch.float32, device=device)  # 优势转张量
        ret_batch = torch.as_tensor(returns, dtype=torch.float32, device=device)  # 回报转张量

         #3. 同一批数据训练多轮              
        policy_loss = torch.tensor(0.0)                              # 占位，后面打印用
        value_loss = torch.tensor(0.0)                              
        entropy_value = torch.tensor(0.0)                            
        approx_kl = 0.0                                              

        for epoch in range(args.n_epochs):                           # 同一批数据训练 n_epochs 轮
            perm = torch.randperm(args.n_steps, device=device)       # 打乱索引，避免 minibatch 有偏

            for start in range(0, args.n_steps, args.minibatch):     # 按 minibatch 切分
                mb = perm[start:start + args.minibatch]              # 取出这一小批的索引

                values, logp, entropy = model.evaluate_actions(      # 用当前策略重新算
                    obs_batch[mb], act_batch[mb]                     # 这批观测和动作
                )                                                    # 返回 V、新 log 概率、熵

                # 和 SB3 对齐
                adv_mb = adv_batch[mb]                               # 取出这一小批的优势
                adv_mb = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)  # 减均值除标准差，加 1e-8 防除零

                ratio = torch.exp(logp - logp_old_batch[mb])  # 新旧策略概率比
                surr1 = ratio * adv_mb                               # 未裁剪的代理目标
                surr2 = (                                            # 裁剪后的代理目标
                    torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip)  # 把比值夹在 [0.8, 1.2]
                    * adv_mb                                         # 再乘优势
                )                                                 # 裁剪结束
                policy_loss = -torch.min(surr1, surr2).mean() # 取两者较小值，加负号变成最小化

                value_loss=F.mse_loss(values, ret_batch[mb])     # Critic 用均方误差拟合回报
                entropy_value = entropy.mean()             # 平均熵，越大越随机

                loss=(                # 总损失
                    policy_loss                    # 策略损失
                    + args.vf_coef * value_loss# 价值损失加权
                    - args.ent_coef * entropy_value      #减熵就是鼓励探索
                )                                                    # 总损失结束

                opt.zero_grad()             # 清空上一步的梯度
                loss.backward()          # 反向传播算梯度
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪防爆炸
                opt.step()                                           # 更新参数

                with torch.no_grad():                                # 只做监控，不需要梯度
                    log_ratio = logp - logp_old_batch[mb]            # log 比值
                    approx_kl = (                                    # 近似 KL 估计
                        (torch.exp(log_ratio) - 1.0) - log_ratio     # 二阶近似式
                    ).mean().item()                                  # 取平均并转成 Python 数字

        if update % eval_every_updates == 0 or update == num_updates:  # 到评估点或最后一次更新
            score = evaluate(                                        # 跑几个完整回合
                model, eval_env, device, episodes=args.eval_episodes  # 用评估环境，跑指定回合数
            )                                                        # 拿到平均回报
            best = max(best, score)                                  # 更新历史最好成绩
            step = update * args.n_steps                             # 换算成累计环境步数
            print(           # 打印一行训练日志
                f"update {update:4d}  steps {step:7d}  "            # 第几次更新、累计步数
                f"eval {score:8.1f}  best {best:8.1f}  "             # 本次成绩、历史最好
                f"pi_loss {policy_loss.item(): .3f}  "               # 策略损失
                f"v_loss {value_loss.item(): .3f}  "                 # 价值损失
                f"entropy {entropy_value.item(): .3f}  "             # 策略熵
                f"kl {approx_kl: .4f}"                               # 近似 KL，太大说明数据快过期
            )                                                        # print 结束
            if args.save_csv:                                        # 需要保存结果
                with open(args.save_csv, "a", newline="", encoding="utf-8") as f:  # 追加模式打开
                    writer = csv.writer(f)                           # 创建写入器
                    writer.writerow([args.label, args.seed, step, f"{score:.6f}"])  # 写一行记录

    print(f"训练结束，最好评估回报：{best:.1f}")                        # 训练结束，报告最好成绩
    env.close()                                                      # 关闭训练环境，释放资源
    eval_env.close()                                                 # 关闭评估环境


if __name__ == "__main__":                                          
    main()                                     