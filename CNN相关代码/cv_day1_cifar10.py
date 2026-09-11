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

#第四段:取出第一批数据进行验证
print("训练样本总数:",len(trainset))
print("10个类别:",trainset.classes)

dataiter=iter(trainloader) #把加载器变成迭代器,相当于加了个书签
images,labels=next(dataiter) #取第一批:images=[8,332,32],labels=[8]
print("一批图片shape",images.shape)
print("一批标签 shape",labels.shape)
print("像素范围:",round(images.min().item(),2),"~",round(images.max().item(),2))
# .item()转换成普通数字

#第五段:看前三张图分别是什么类别
for i in range(3):
    label_idx=labels[i].item()
    print(f"第{i}张:标签索引{label_idx} ->"
            f"{trainset.classes[label_idx]}"
          )

#可视化一批图
images,labels=next(iter(trainloader))#再取一批
fig,axes=plt.subplots(2,4,figsize=(10,5))

for i,ax in enumerate(axes.flat):
    img=images[i].permute(1,2,0)*0.5+0.5
    ax.imshow(img.clamp(0,1)) #clamp保证范围0到1
    ax.set_title(trainset.classes[labels[i]]) #标题等于类别名
    ax.axis('off')
plt.show()