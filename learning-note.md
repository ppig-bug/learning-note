# 强化学习的数学原理
    入门了解：强化学习的最终目的就是要求解最优策略
### 基本概念
辅助理解的一个模型：grid-world example,如下：
| s1 | s2 | s3 |
|----|----|----|
| s4 | s5 | **<span style="background-color:#ffbc00">s6</span>** |
| **<span style="background-color:#ffbc00">s7</span>** | s8 | **<span style="background-color:#40e0f0">s9</span>** |

1.State:agent相对于环境的状态，在grid就是location，即s1,s2等

2.state space:本质就是一个集合 S={si}

3.Action:每个状态对应5个可能的动作。
a1向上 a2向右 a3向下 a4向左 a5不动

4.Action Space of a state:同理也是一个集合

5.State transition ：实际上是定义agent和环境的交互行为
举例：s1通过a2到s2,s1通过a1到s1

6.Forbidden area:(黄色)
situation1：accessible but with penalty
situation2: inaccessible

7.State transition probability:
用条件概率描述state transition:
<span style="color:red;">
**p(s2|s1,a2)=1**
**p(si|s1,a2)=0 任意i!=2**
</span>

8.Policy:在某个状态采取怎样的action

9.Mathematical representation:(强化学习中Π统一指的是策略)（确定性）

Π（a1|s1）=0

Π（a2|s1）=1

Π（a3|s1）=0

Π（a4|s1）=0

Π（a5|s1）=0

10.stochastic policies:(不确定性)：

Π（a1|s1）=0

Π（a2|s1）=0.5

Π（a3|s1）=0.5

Π（a4|s1）=0

Π（a5|s1）=0

11.reward：一个标量，在agent采取动作后得到一个数。正数即鼓励，负数即惩罚，为0可以相当于鼓励
<span style="color:red;">
**p(r=-1|s1,a1)=1 and p(r!=-1|s1,a1)=0**
</span>

12.trajectory:state-action-reward chain

13.return:trajectory所有的reward相加

14.Discounted return  :为解决return无限加数得到无穷值的情况，引入discount rate γ，0<=γ<=1

    discounted return=0+γ0+γ<sup>2</sup>0+...=γ<sup>3</sup>(1/(1-γ))
γ趋向于0，return越依赖于前面
γ趋向于1，return越依赖后面

15.Episode:一个有限步的trajectory

### Markov decision process(MDP)

1.key elements of MDP:

    sets:state,action,reward
    Probability distribution:state transition probability,reward probability
    Policy:Π（a|s)

2.Markov property:
    
    p(St+1|=a t+1,St,...a1,s0)=p(St+1|=a t+1,St)
    p(r t+1|=a t+1,St,...a1,s0)=p(r t+1|=a t+1,St)
