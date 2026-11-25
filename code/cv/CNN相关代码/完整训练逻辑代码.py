import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader,Subset#子集
import torchvision.transforms as T
import matplotlib.pyplot as plt

#1.准备数据
device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用设备：", device)


torch.manual_seed(42)
DATAROOT=r"C:\Users\17783\Desktop\暑期学习\data"

transform=T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
]
)

trainset=torchvision.datasets.CIFAR10(root=DATAROOT,train=True,download=True,transform=transform)
testset=torchvision.datasets.CIFAR10(root=DATAROOT,train=False,download=True,transform=transform)

trainset=Subset(trainset,range(5000))

trainloader=DataLoader(trainset,batch_size=64,shuffle=True,num_workers=0)
testloader=DataLoader(testset,batch_size=256,shuffle=False,num_workers=0)

print("训练集：",len(trainset),"测试集：",len(testset))

#2.模型model
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.Linear(32*8*8,10),
        )
    def forward(self,x):
        return self.classifier(self.features(x))

model=TinyCNN().to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

#3.训练
EPOCHS=5
train_losses,val_accs=[],[]

for epoch in range(EPOCHS):
    model.train()
    running_loss=0.0

    for images,labels,in trainloader:
        images,labels=images.to(device),labels.to(device)

        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*images.size(0)

    train_loss=running_loss/len(trainset)
    train_losses.append(train_loss) 

        # 验证阶段
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    val_accs.append(val_acc)                 
    print(f"Epoch {epoch+1}/{EPOCHS}  train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
    
#4.画曲线
fig,ax1=plt.subplots(figsize=(8,4)) #ax1是一套坐标体系
ax1.plot(train_losses,"b-o",label='train loss')#一般x轴默认是0123，y轴是train_loss
ax1.set_xlabel('epoch')
ax1.set_ylabel('loss', color='b')

ax2 = ax1.twinx()#twin意思是和ax1共用一个x轴，自己独立一个y轴
ax2.plot(val_accs, 'r-s', label='val acc')
ax2.set_ylabel('accuracy', color='r')

plt.title('CIFAR-10 TinyCNN')
plt.show()

# 5. 保存模型 
torch.save(model.state_dict(), "cnn_cifar10.pth")
print("已保存 cnn_cifar10.pth")
