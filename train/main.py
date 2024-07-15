import torch
from torch import nn, device
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pac.xyc_fuc as xyc

data=np.load('../data/MNIST/mnist.npz')
x_train=torch.tensor(data['x_train'],dtype=torch.float).reshape(-1,1,28,28)
y_train=torch.tensor(data['y_train'],dtype=torch.int64)
x_test=torch.tensor(data['x_test'],dtype=torch.float).reshape(-1,1,28,28)
y_test=torch.tensor(data['y_test'],dtype=torch.int64)


net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),nn.Sigmoid(),nn.AvgPool2d(kernel_size=2,stride=2)
                  ,nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten(),nn.Linear(16*5*5,120)
                  ,nn.Linear(120,84),nn.Sigmoid(),nn.Linear(84,10))

lr=0.1
batch_size=256
num_epoch=50
train_iter=xyc.data_iter(xyc.My_Datasets((x_train,y_train)),batch_size,shuffle=True)
test_iter=xyc.data_iter(xyc.My_Datasets([x_test,y_test]),batch_size,shuffle=True)
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD
trainer=xyc.Train(net,train_iter,test_iter,loss,optimizer,num_epoch,lr,xyc.try_gpu())
trainer.train()
trainer.save_stats('../Model/model.pt')