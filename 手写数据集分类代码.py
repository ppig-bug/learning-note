import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

#1.准备数据
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,)),
])

train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transform)

train_loader=DataLoader(train_dataset,batch_size=64,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=256,shuffle=False)

#2.定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)#10个数字类别

    def forward(self,x):
        x=x.view(x.size(0),-1) #把28*28的图拉成784个像素一排，-1让pyTorch自动算这一排有多长
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        return self.fc3(x)

model=MLP()

#3.损失函数
criterion=nn.CrossEntropyLoss()

#4.优化器
optimizer=optim.Adam(model.parameters(),lr=0.001)

#5.训练循环
EPOCHS=5
for epoch in range(EPOCHS):
    model.train()
    total_loss=0.0
    for images,labels in train_loader:
        out=model(images)
        loss=criterion(out,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*images.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS}  平均loss={total_loss/len(train_dataset):.4f}")


#6.评估
model.eval()
correct,total=0,0
with torch.no_grad():
    for images,labels in test_loader:
        out=model(images)
        pred=out.argmax(dim=1)
        correct+=(pred==labels).sum().item()
        total+=labels.size(0)
print(f'测试集准确率：{correct/total*100:.2f}%')