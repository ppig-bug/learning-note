import numpy as np
import torch
import torch.nn as nn

#1.手写最大池化
def max_pool2d_numpy(x,k=2,stride=None):#x就是numpy数组[H,W],一般默认步长等于k
    if stride is None:
        stride=k
    H,W=x.shape
    out_H=(H-k)//stride+1
    out_W=(W-k)//stride+1
    out=np.zeros((out_H,out_W))
    for i in range(out_H):#向下滑动
        for j in range(out_W):#向右滑动
            window=x[i*stride:i*stride+k,j*stride:j*stride+k]
            out[i,j]=window.max()
    return out


#2.手写平均池化
def avg_pool2d_numpy(x,k=2,stride=None):#x就是numpy数组[H,W],一般默认步长等于k
    if stride is None:
        stride=k
    H,W=x.shape
    out_H=(H-k)//stride+1
    out_W=(W-k)//stride+1
    out=np.zeros((out_H,out_W))
    for i in range(out_H):#向下滑动
        for j in range(out_W):#向右滑动
            window=x[i*stride:i*stride+k,j*stride:j*stride+k]
            out[i,j]=window.mean()
    return out

#把同一份数据丢给手写函数和框架各自跑一下

x=np.array(
    [[1,2,3,4],
    [5,6,7,8],
    [9,10,11,12],
    [13,14,15,16]],dtype=float
)

t=torch.tensor(x,dtype=torch.float32).view(1,1,4,4) #转化成框架需要的[N,C,H,W]思维

手写_max=max_pool2d_numpy(x,2)
手写_avg=avg_pool2d_numpy(x,2)
框架_max=nn.MaxPool2d(2)(t).squeeze().numpy() #.squeeze().numpy()作用是去掉多余的维度，再转回numpy才可以比较
框架_avg=nn.MaxPool2d(2)(t).squeeze().numpy()

print("手写最大池化：\n", 手写_max)
print("框架最大池化：\n", 框架_max)
print("手写平均池化：\n", 手写_avg)
print("框架平均池化：\n", 框架_avg)

#最后用np.allclose()比结果
print("\n最大池化 手写 == 框架 ?", np.allclose(手写_max, 框架_max))
print("平均池化 手写 == 框架 ?", np.allclose(手写_avg, 框架_avg))

    


