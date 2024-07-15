import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch import nn


def accuracy(y_pred, y_true, one_hot=False):
    if one_hot:
        y_true = torch.argmax(y_true, dim=1)
        # print(y_true)
    y_pred = torch.argmax(y_pred, dim=1)
    arr = torch.zeros_like(y_pred)
    arr[y_pred == y_true] = 1
    acc = arr.sum() / arr.shape[0]
    # print(acc)
    return acc


# accuracy(torch.tensor([[0,0.5,0.3],[1,0,0],[1,0,0]]),torch.tensor([0,0,0]),0)

def plot(x, y, figsize=(10, 10), xlim=(0, 1), ylim=(0, 1), label=None, title=None):
    plt.figure(figsize=figsize, dpi=100)
    plt.xlim(xlim)
    plt.ylim(ylim)

    if hasattr(y, 'shape'):
        for i in range(y.shape[0]):
            plt.plot(x, y[i], label=label[i])
    elif hasattr(y, '__len__'):
        for i in range(len(y)):
            plt.plot(x, y[i], label=label[i])

    plt.legend(loc='best')
    plt.grid()
    plt.title(title)
    plt.show()


class Timer:  # @save
    """记录多次运行时间"""

    def __init__(self):
        self.tik = None
        self.times = []

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Train:
    """
    针对datasets,dataloader的训练器
    发现交叉熵支持编码和独热编码
     example:
    LOSS=nn.CrossEntropyLoss()
    OPTIMIZER=torch.optim.SGD
    trainer=Train(net,train_iter,test_iter,LOSS,OPTIMIZER,num_epoch,lr,device='cuda',initial=True,one_hot=False)
    trainer.train()

    """

    def __init__(self, model, train_iter, test_iter, loss, optimizer, epochs, lr, device='cpu', initial=True,
                 one_hot=False):
        self.model = model
        self.loss = loss

        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.epochs = epochs
        self.device = device
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.one_hot = one_hot
        if initial:
            self.init_model()
        self.model.to(self.device)

        print('training on', device)

    def info(self):
        print(self.model)

    def init_model(self, func=nn.init.xavier_uniform_):
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv2d:
                func(m.weight)

        self.model.apply(init_weights)

    def train(self):
        timer = Timer()
        test_acc = []
        train_acc = []
        los = []
        for epoch in range(self.epochs):
            timer.start()
            self.model.train()
            loss_count = []
            acc_train = []
            acc_test = []
            for (X, y) in self.train_iter:
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_pred = self.model(X)
                acc_train.append(accuracy(y_pred, y, one_hot=self.one_hot))
                loss = self.loss(y_pred, y)
                loss_count.append(loss)
                loss.backward()
                self.optimizer.step()

            los.append((sum(loss_count) / len(loss_count)).detach().cpu().numpy())
            train_acc.append((sum(acc_train) / len(acc_train)).to('cpu'))
            timer.stop()
            self.model.eval()
            with torch.no_grad():
                for (X, y) in self.test_iter:
                    X, y = X.to(self.device), y.to(self.device)
                    y_pred = self.model(X)
                    acc_test.append(accuracy(y_pred, y, one_hot=self.one_hot).to('cpu'))
            test_acc.append(sum(acc_test) / len(acc_test))
            print(
                f'epoch: {epoch}, test_acc: {test_acc[epoch].item():.3f},train_acc: {train_acc[epoch].item():.3f},loss: {los[epoch].item():.3f},time: {timer.times[epoch]:.3f}')
        print(f'time_sum, {timer.sum():.4f},time_avg: {timer.avg():.3f}')

        plot([epoch for epoch in range(self.epochs)], [train_acc, test_acc, los],
             label=['train_acc', 'test_acc', 'loss'], title='train_plot', xlim=(0, self.epochs), ylim=[0, 1.1])
        return [train_acc, test_acc, los]

    def save_stats(self, path):
        """save.pt/ckpt/pth/pkl"""
        torch.save(self.model.state_dict(), path)


class My_Datasets(torch.utils.data.Dataset):
    """
    传入tuple/list/tensor [(tensor/np,tensor/np)]
    返回datasets
    """

    def __init__(self, data, transform=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.x = data[0]
        if data[0].shape[0] != data[1].shape[0]:
            print('attention data[0] cannot match data[1]')
        else:
            print('Shape:{}'.format(data[0].shape[0]))

    def __len__(self):
        #重写
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.data[0][idx], self.data[1][idx]

        if self.transform:
            sample = self.transform(sample[0]), self.transform(sample[1])
        return sample


def data_iter(datasets, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=shuffle)


def try_gpu():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')





'''torch_cv.py'''
# import torch
# from torch import nn, device
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
#
# data=np.load('../data/MNIST/mnist.npz')
# x_train=torch.tensor(data['x_train'],dtype=torch.float).reshape(-1,1,28,28)
# y_train=torch.tensor(data['y_train'],dtype=torch.int64)
# x_test=torch.tensor(data['x_test'],dtype=torch.float).reshape(-1,1,28,28)
# y_test=torch.tensor(data['y_test'],dtype=torch.int64)
#
#
# net=nn.Sequential(nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),nn.Sigmoid(),nn.AvgPool2d(kernel_size=2,stride=2)
#                   ,nn.Conv2d(6,16,kernel_size=5),nn.Sigmoid(),nn.MaxPool2d(kernel_size=2,stride=2),nn.Flatten(),nn.Linear(16*5*5,120)
#                   ,nn.Linear(120,84),nn.Sigmoid(),nn.Linear(84,10))
#
#
# lr=0.1
# batch_size=256
# num_epoch=5
#
#
#
# train_iter=data_iter(My_Datasets([x_train,y_train]),batch_size=batch_size,shuffle=True)
# test_iter=data_iter(My_Datasets((x_test, y_test)), batch_size=batch_size, shuffle=False)
#
# LOSS=nn.CrossEntropyLoss()
# OPTIMIZER=torch.optim.SGD
# trainer=Train(net,train_iter,test_iter,LOSS,OPTIMIZER,num_epoch,lr,device='cuda',initial=True,one_hot=False)
# trainer.train()
