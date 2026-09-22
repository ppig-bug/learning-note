#持续更新中
import torch
import torch.nn as nn
import math

#定义自注意力
class SelfAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout) #对10%的神经元做一个神经失活，防止过拟合
        self.softmax=nn.SoftMax(dim=-1) #将得分转化成概率分布，在最后一个维度进行

    def forward(self,Q,K,V,mask=None):
        #X:batch,seq_len,d_model 一次送到模型的个数；序列长度即一个样本中的token数量；embedding向量的维度
        # Q的向量维度：batch,heads,seq_len_q,d_k
        # K的向量维度 batch,heads,seq_len_k,d_k
        # V的向量维度 batch,heads,seq_len_v,d_v
        d_k=Q.size(-1) #q的最后一维是每个query向量的维度，代表我们对每个query进行缩放
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k) #K.transpose(-2,-1)表示K的转置 
        #如果提供了mask,则通过mask==0来找到需要屏蔽的位置，masked_fill()会将这些值改成-inf负无穷
        #经过softmax之后这些位置的值会变成0，代表被忽略
        #设置mask==0,表示被屏蔽；mask==1代表该位置可见
        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))
        #对key进行softmax,得到注意力权重矩阵，对每一个query的key权重之和为1
        attn=self.softmax(scores)
        attn=self.dropout(attn)#对注意力权重进行dropout,防止过拟合
        out=torch.matmul(attn,V)
        return out,attn

# 定义多头注意力
class MultiHeadAttention(nn.Module):    
    def __init__(self,d_model,n_heads,dropout=0.1):
        super().__init__()
        #embedding的维度512需要被头数8整除
        assert d_model % n_heads==0
        self.d_k=d_model//n_heads #每个头的维度
        self.n_heads=n_heads

        #将输入映射到Q，K，V三个向量,通过线性映射让模型更具有学习力
        self.W_q=nn.Linear(d_model,d_model)# 注意：维度不需要改变
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.fc=nn.Linear(d_model,d_model) #多头拼接

        self.attention=SelfAttention(dropout)
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(d_model) #用于残差后的1归一化

    def forward(self,q,k,v,mask=None):
        batch_size=q.size(0) #获取batch的大小
        # q的维度batch,seq_len,d_model ->batch,seq_len,self.n_heads,self.d_k ->batch,self.n_heads,seq_len,self.d_k
        #交换一二维度的目的：为了让每个注意力头独立处理整个序列，方便后续计算注意力权重
        Q=self.W_q(q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        K=self.W_k(k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
        V=self.W_v(v).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)

        #计算注意力
        out,attn = self.attention(Q,K,V,mask) #attn为注意力权重，out为注意力加权后的值
        #contiguous目的：让tensor在内存里面存储，避免view的时候产生报错
        out=out.tranpose(1,2).contiguous().view(batch_size,-1.self.n_heads*self.d_k)
        out=self.fc(out) #让输入和输出一致，方便残差连接
        out=self.dropout(out)
        #残差连接+layernorm
        return self.norm(out+q),attn #返回输出和注意力权重
