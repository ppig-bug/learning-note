#未完成
import torch
import torch.nn as nn

model=nn.Sequential(nn.Flatten(),nn.Linear(3*32*32,10)) #10是类别数

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)

images=torch.randn(8,3,32,32)
labels=torch.randint(0,10,(8,))

optimizer.zero_grad()
outputs=model(images)
loss=criterion(outputs,labels)
loss.backward()
optimizer.step()

print("loss=",loss.item())
