import cv2
import numpy as np
import torch
import torch.nn as nn
import pac.xyc_fuc as xyc

net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),nn.Sigmoid(),nn.AvgPool2d(kernel_size=2,stride=2)
                  ,nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten(),nn.Linear(16*5*5,120)
                  ,nn.Linear(120,84),nn.Sigmoid(),nn.Linear(84,10))

net.load_state_dict(torch.load("../Model/model.pt"))
net.eval()
im=cv2.imread("../data/num0-9/1.jpg",0)
im=cv2.resize(im,(40,40))
im=im[6:34,6:34]
cv2.imshow("image",im)
print(torch.argmax(net(torch.tensor(im).float().reshape(-1,1,28,28))))
cv2.waitKey(0)
cv2.destroyAllWindows()

