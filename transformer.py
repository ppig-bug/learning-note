import torch
import torch.nn as nn
import math

#定义自注意力
class SelfAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super().__init__()
        self.dropout=nn.Dropout(dropout) #对10%的神经元做一个神经失活，防止过拟合
        self.softmax=nn.Softmax(dim=-1) #将得分转化成概率分布，在最后一个维度进行

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
        out=out.transpose(1,2).contiguous().view(batch_size,-1,self.n_heads*self.d_k)
        out=self.fc(out) #让输入和输出一致，方便残差连接
        out=self.dropout(out)
        #残差连接+layernorm
        return self.norm(out+q),attn #返回输出和注意力权重

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super().__init__()
        self.fc1=nn.Linear(d_model,d_ff) #输入维度为d_model,输出为d_ff ,为了让模型学到一个更丰富的特征
        self.fc2=nn.Linear(d_ff,d_model) #保证第二个线性层输出维度等于第一个线性层的输入维度，为了后续残差链接
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(d_model) #对最后一个维度进行归一化

    def forward(self,x):
        out=self.fc2(self.dropout(torch.relu(self.fc1(x)))) #先经过第一个线性层，再经过relu,在经过dropout,再经过第二个线性层
        return self.norm(out+x) #先经过残差连接，再经过归一化

class EncoderLayer(nn.Module):
    def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
        super().__init__()
        # 多头注意力机制 输入为src，实现训练内部的信息交互，每个token都可以看到序列中其他的token，从而学习上下文信息
        self.self_attn=MultiHeadAttention(d_model,n_heads,dropout)
        #对每个位置向量独立进行非线性变换，可以提升模型表达能力
        self.ffn=FeedForward(d_model,d_ff,dropout)

    def forward(self,src,src_mask=None):
        # src 输入序列张量，形状batch,seq_len,d_model
        #src_mask 屏蔽padding的位置，避免模型关注无效的token
        out,_=self.self_attn(src,src,src,src_mask)
        # 经过前馈神经网络，每个位置的token都会单独通过两层线性层映射和激活函数，来提升模型的表达能力
        out=self.ffn(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # 形状 [1, max_len, d_model]，不参与训练

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])   # 只取和输入一样长的那一段

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model    = d_model
        self.embedding  = nn.Embedding(vocab_size, d_model)      # token id → 向量
        self.pos        = PositionalEncoding(d_model, max_len, dropout)
        self.layers     = nn.ModuleList(                         # 堆 num_layers 层
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask=None):
        x = self.embedding(src) * math.sqrt(self.d_model)   # 乘 sqrt(d_model) 是原论文的做法
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
     def __init__(self,d_model,n_heads,d_ff,dropout=0.1):
            super().__init__()
            #Mask多头注意力机制
            # 输入tgt(目标序列) 在翻译任务中 已经生成的前几个单词
            # 计算目标序列内部的自注意力，通过mask挡住未来的token
            self.self_attn=MultiHeadAttention(d_model,n_heads,dropout)
            #交叉注意力，和encoder做交互
            #输入Q=当前解码器的输出，K=V=来自编码器的memory
            #目的：将目标序列与原序列对齐
            self.cross_attn=MultiHeadAttention(d_model,n_heads,dropout)
            self.ffn=FeedForward(d_model,d_ff,dropout)

     def forward(self,tgt,memory,tgt_mask=None,memory_mask=None):
            #tgt目标序列 memory:编码器的输出（原序列的表示）
            #tgt_mask,屏蔽未来的token
            out,_=self.self_attn(tgt,tgt,tgt,tgt_mask)
            #与原序列进行交互，Q解码器当前的输出out,K=V=memory
            out,_=self.cross_attn(out,memory,memory,memory_mask)
            out=self.ffn(out)
            return out

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model   = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos       = PositionalEncoding(d_model, max_len, dropout)
        self.layers    = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,#原语言词表的大小
                 tgt_vocab,#目标语言词表大小
                 d_model=512,#embedding向量的维度
                 n_heads=8,#多头注意力的头数
                 num_encoder_layers=6, #编码器的层数
                 num_decoder_layers=6,#解码器的层数
                 d_ff=2048, #FFN隐藏层维度
                 dropout=0.1,#丢弃比例
                 max_len=5000): #最大序列的长度
        super().__init__()

        self.encoder= Encoder(
            src_vocab,d_model,n_heads,num_encoder_layers,d_ff,dropout,max_len
        )

        self.decoder = Decoder(tgt_vocab, d_model, n_heads, num_decoder_layers, d_ff, dropout, max_len)
        self.fc_out  = nn.Linear(d_model, tgt_vocab) 

    def forward(self,src,tgt,src_mask=None,tgt_mask=None,memory_mask=None):
        # 编码器前向传播
        memory=self.encoder(src,src_mask)
        # 解码器前向传播
        out=self.decoder(tgt,memory,tgt_mask,memory_mask)
        # 返回transformer输出batch，seq_len_tgt,tgt_vocab
        return self.fc_out(out)

def generate_mask(size):#size是序列长度
    # torch.triu(torch.ones(size,size),diagonal=1) 会生成上三角，不含对角线
    mask=torch.triu(torch.ones(size,size),diagonal=1).bool()
    # 这样做是为了明确生成了上三角（需要屏蔽的位置），然后通过mask==0得到可见的部分
    return mask==0 #True可见，False屏蔽

src_vocab=10000
tgt_vocab=10000
# 初始化模型
model=Transformer(src_vocab,tgt_vocab)
src=torch.randint(0,src_vocab,(32,10)) #原序列batch=32,src_len=10 每个元素是token ID
# tgt.size(1) 取目标序列长度
tgt = torch.randint(0, tgt_vocab, (32, 8))
tgt_mask=generate_mask(tgt.size(1)).to(tgt.device)
out=model(src,tgt,tgt_mask=tgt_mask) #前向传播
# 每个目标token 对应词表中每个词的预测概率
print(out.shape) #batch,tgt_len,tgt_vocab
