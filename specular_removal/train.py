# coding:utf-8
import os
import time
from datetime import datetime

import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .dataset import SRDataset
from .network import SRNet


def exception_handler(train_func):
    """ 处理训练过程中发生的异常并保存模型 """
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            return train_func(train_pipeline, *args, **kwargs)
        except KeyboardInterrupt:
            train_pipeline.save()
            exit()
    return wrapper


class SRNetLoss(nn.Module):
    """ SRNet 的损失函数 """

    def __init__(self):
        super().__init__()

    def forward(self, M_hat, M, S_hat, S, D_hat, D):
        loss = F.mse_loss(D_hat, D) + F.mse_loss(S_hat, S) + \
            F.binary_cross_entropy(M_hat, M)
        return loss


class TrainPipeline:
    """ 训练流水线 """

    def __init__(self, train_dataset_dir: str, test_dataset_dir: str, lr=0.01, step_size=10,
                 train_batch_size=10, test_batch_size=10, epochs=20, test_freq=5, use_gpu=True,
                 model_dir=None):
        """
        Parameters
        ----------
        train_dataset_dir: str
            训练集文件夹路径

        test_dataset_dir: str
            测试集文件夹路径

        lr: float
            学习率

        step_size: int
            学习率衰减的的步长

        train_batch_size: int
            训练集 batch 大小

        test_batch_size: int
            测试集 batch 大小

        epochs: int
            世代数

        test_freq: int
            测试频率

        use_gpu: bool
            是否使用 GPU

        model_dir: str
            模型保存文件夹路径，如果为 None，则保存到 `'./model'`
        """
        self.lr = lr
        self.epochs = epochs
        self.test_freq = test_freq
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.model = SRNet().to(self.device)
        self.model_dir = model_dir if model_dir else 'model'
        # 创建数据集和数据加载器
        self.train_dataset = SRDataset(train_dataset_dir)
        self.test_dataset = SRDataset(test_dataset_dir)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=test_batch_size, shuffle=True)
        # 定义优化器和损失函数
        self.criterion = SRNetLoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.lr_scheduler = StepLR(
            optimizer=self.optimizer, step_size=step_size, gamma=0.1)

    def save(self):
        """ 保存模型 """
        os.makedirs(self.model_dir, exist_ok=True)
        t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
        path = f'{self.model_dir}/SRNet_{t}.pth'
        self.model.eval()
        torch.save(self.model.state_dict(), path)
        print(f'🎉 已将当前模型保存到 {os.path.join(os.getcwd(), path)}')

    @exception_handler
    def train(self):
        """ 训练模型 """
        train_losses = []
        test_losses = []
        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        print('🚀 开始训练！')
        for e in range(self.epochs):
            with tqdm(self.train_loader, bar_format=bar_format) as train_bar:
                train_bar.set_description(f"\33[36m🌌 Epoch {e + 1:3d}")
                start_time = datetime.now()
                self.model.train()
                for I, M, S, D in self.train_loader:
                    I = I.to(self.device)
                    M = M.to(self.device)
                    S = S.to(self.device)
                    D = D.to(self.device)
                    M_hat, S_hat, D_hat = self.model(I)
                    self.optimizer.zero_grad()
                    train_loss = self.criterion(M_hat, M, S_hat, S, D_hat, D)
                    train_loss.backward()
                    self.optimizer.step()
                    cost_time = datetime.now() - start_time
                    train_bar.set_postfix_str(
                        f'训练损失：{train_loss.item():.5f}, 执行时间：{cost_time}\33[0m')
                    train_bar.update()

            # 测试模型
            if (e+1) % self.test_freq == 0:
                with tqdm(self.test_loader, bar_format=bar_format) as test_bar:
                    test_bar.set_description('\33[35m🛸 测试中')
                    start_time = datetime.now()
                    self.model.eval()
                    for I, M, S, D in self.test_loader:
                        I = I.to(self.device)
                        M = M.to(self.device)
                        S = S.to(self.device)
                        D = D.to(self.device)
                        M_hat, S_hat, D_hat = self.model(I)
                        test_loss = self.criterion(M_hat, M, S_hat, S, D_hat, D)
                        cost_time = datetime.now() - start_time
                        test_bar.set_postfix_str(
                            f'测试损失：{test_loss.item():.5f}, 执行时间：{cost_time}\33[0m')
                        test_bar.update()

                test_losses.append(test_loss.item())
                self.save()

            # 记录误差
            train_loss = train_loss.item()
            train_losses.append(train_loss)
            self.lr_scheduler.step()

        return train_losses, test_losses
