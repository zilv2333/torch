import torch
from d2l import torch as d2l

"""
d2l部分缺失的包，——ch3
"""
class ch3():
    def __init__(self):
        self.intro='d2l-ch3'
    def train_ch3(self,net, train_iter, test_iter, num_epochs, loss, updater):
        animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylabel='loss', ylim=[0.3, 0.9],
                                legend=['train loss', 'train acc', 'text acc'])
        for epoch in range(num_epochs):
            train_metric = self.train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = self.evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metric + (test_acc,))


    def train_epoch_ch3(self,net, train_iter, loss, updater):
        """updater优化器，传入批量数batch，w,b为全局变量"""
        if isinstance(net, torch.nn.Module):
            net.train()
        # 损失和，精度，总数
        metric = d2l.Accumulator(3)
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 内置优化器
                updater.zero_grad()
                l.mean().backward()
                updater.step()
            else:
                # 自定义优化器
                l.sum().backward()
                updater(X.shape[0])
            metric.add(float(l.sum()), d2l.accuracy(y_hat, y), y.numel())
        return metric[0] / metric[2], metric[1] / metric[2]


    def evaluate_accuracy(self,net, data_iter):  # @save
        """计算在指定数据集上模型的精度"""
        if isinstance(net, torch.nn.Module):
            net.eval()  # 将模型设置为评估模式
        metric = d2l.Accumulator(2)  # 正确预测数、预测总数
        with torch.no_grad():
            for X, y in data_iter:
                metric.add(d2l.accuracy(net(X), y), y.numel())

        return metric[0] / metric[1]

    def predict_ch3(self,net,test_iter,n=6):
        for X,y in test_iter:
            break
        trues=d2l.get_fashion_mnist_labels(y)
        preds=d2l.get_fashion_mnist_labels(net(X).argmax(dim=1))
        title=[true+'\n'+pred for true,pred in zip(trues,preds)]
        d2l.show_images(X[0:n].reshape((n,28,28)),1,n,titles=title[0:n])


