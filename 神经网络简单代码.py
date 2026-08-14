import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(42)

#1.生成训练数据集
N=2000
x=torch.rand(N,2) 
#2000个维点，每个坐标在[0,1）
y=(x[:,0]**2+x[:,1]**2<0.25).long()
print("输入形状：",x.shape,"标签形状：",y.shape)

#2.定义模型，用module
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(2,16)
        self.fc2=nn.Linear(16,8)
        self.fc3=nn.Linear(8,2)

    def forward(self,x):
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)

model=MLP()
#打印网格结构
print("网格结构：",model)

#3.损失函数以及优化器
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)#Adam:自适应学习率优化算法


#4.训练循环
EPOCHS=200
for epoch in range(EPOCHS):
    out=model(x)
    loss=criterion(out,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if(epoch+1)%20==0:
        print(f"Epoch {epoch+1:3d} loss={loss.item():.4f}")


#评估(不构建梯度)
with torch.no_grad():
    pred=model(x).argmax(dim=1) #取分数最大的类别作为预测结果
    acc=(pred==y).float().mean().item()
print(f"训练集的准确率：{acc*100:.1f}%")