
import torch.nn as nn
import torch
import torchvision #视觉库,自带数据集
import torchvision.transforms as T #预处理工具包,Compose,ToTensor,Normalize
from torch.utils.data import DataLoader #DataLoader:分批/打乱
import matplotlib.pyplot as plt

#第一段:预处理流水线
transform=T.Compose([ #Compose 把多个处理串成一条线
    T.ToTensor(),
    T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))#三个通道每个通道的均值,标准差都取0.5
]
)

#第二段:创建Dataset(只负责第i个样本+它的标签)
trainset=torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform,#每次取一个样本,先走上面的预处理
) #返回处理后的图,标签+数字

#第三段:创建DataLoader(负责"分批,打乱,自动取数)
trainloader=DataLoader(trainset,
    batch_size=16,
    shuffle=True,
    num_workers=0,#取数,把图片从硬盘里面读出来,经过一系列操作,再递给模型使用,等于0时就是什么都是自己干,等于2时就是有个帮手帮你
)


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,padding=1),#参数分别的意思是，输入有几个通道，输出有几个通道，窗户大小3*3，padding就是四周各补一圈0
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,kernel_size=3,padding=1),#参数分别的意思是，输入有几个通道，输出有几个通道，窗户大小3*3，padding就是四周各补一圈0
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier=nn.Sequential(
            nn.Flatten(),#展品成向量
            nn.Linear(32*8*8,10)#10是类别数
        )

    def forward(self,x):
        return self.classifier(self.features(x))

model=TinyCNN()


images, labels = next(iter(trainloader))   # [8, 3, 32, 32]
out = model(images)
print("输入:", images.shape)
print("输出:", out.shape)       