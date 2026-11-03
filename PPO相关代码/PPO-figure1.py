#复现ppo论文里面的figure1
#内容是clipped surrogate objective随概率比ratio的变化
#论文中的公式：L^CLIP = E[ min( r*A, clip(r, 1-eps, 1+eps)*A ) ]

import os #用于建result目录
import numpy as np
import matplotlib
matplotlib.use("Agg")#这个是用无界面的后端，直接传文件
import matplotlib.pyplot as plt

epsilon=0.2
r=np.linspace(0.0,2.0,1001) #linspace是在区间里面均匀的取点

def compute_clip_obj(r,A,esp=epsilon):
    return np.minimum(A*r,np.clip(r,1.0-esp,1.0+esp)*A)

def kl_penalty_obj(r,A,beta):#拿kl做对照组
    kl=-np.log(np.maximum(r,1e-12))+(r-1.0) #kl的二阶近似，加上1e-12是为了防止log0
    return r*A-beta*kl

fig,axes=plt.subplots(1,2,figsize=(11,4.2)) #一行两列做A是否大于0的区分


for ax,A in zip(axes,[1.0,-1.0]):#依次画A=1和A=-1的子图
    ax.plot(r, r * A, "--", color="0.55", lw=1.5, label=r"$r\,A$ (unclipped)")  # 灰色虚线：没有裁剪时的目标
    ax.plot(r, compute_clip_obj(r, A), color="#1f77b4", lw=2.8, label=r"$L^{CLIP}$")    # 蓝色实线：裁剪后的目标
    
    ax.axvline(1.0-epsilon,color="k",ls="--",lw=0.9,alpha=0.7) # 竖线标出下界 1-epsilon = 0.8
    ax.axvline(1.0+epsilon,color="k",ls="--",lw=0.9,alpha=0.7) # 竖线标出下界 1+epsilon =1.2
    ax.axvline(1.0,color="k",ls="-",lw=0.8,alpha=0.5) # 竖线标出r=1,就是新旧策略相同
    ax.axhline(0.0,color="k",lw=0.8,alpha=0.5) # 水平零线

    ax.plot(1.0, A, "o", color="tab:red", ms=8, zorder=5,
            label=r"starting point $r=1$") # 论文里的红点：优化开始的位置

    # 手动锁死纵轴：既要装下 r*A（到 ±2），又不让 L^CLIP 被压扁
    if A > 0:
        ax.set_ylim(-0.05, 2.1)
    else:
        ax.set_ylim(-2.1, 0.05)

    ytxt = 0.10 if A > 0 else -1.95 # 标签放在底部空白处，不压曲线
    ax.text(1.0 -epsilon - 0.13, ytxt, r"$1-\epsilon$", fontsize=9) # 标注下界标签
    ax.text(1.0 +epsilon + 0.02, ytxt, r"$1+\epsilon$", fontsize=9)  # 标注上界标签

    ax.set_xlabel("probability ratio  r")                           
    ax.set_ylabel("surrogate objective")                            
    ax.set_title("A > 0" if A > 0 else "A < 0")                
    ax.grid(alpha=0.3)                                              
    ax.legend(loc="best", fontsize=8)     

plt.suptitle(r"Figure 1 reproduction: clipped surrogate objective ($\epsilon=0.2$)", y=1.02)  # 总标题
plt.tight_layout()#自动调整
os.makedirs("results", exist_ok=True)     # 确保输出目录存在
out = "results/fig1_clip.png"  # 输出文件名
plt.savefig(out, dpi=150, bbox_inches="tight")  #存图，150 分辨率，裁掉多余白边
print("saved:", out)   # 终端提示保存路径