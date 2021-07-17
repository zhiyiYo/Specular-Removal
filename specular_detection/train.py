# coding:utf-8
import os
import time
from datetime import datetime

import torch
from torch import cuda
from torch.nn import BCELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import SDDataset
from network import SDNet


def save_model(train_func):
    """ 保存模型 """
    def wrapper(train_pipeline, *args, **kwargs):
        try:
            train_func(train_pipeline, *args, **kwargs)
        except KeyboardInterrupt:
            os.makedirs('../model', exist_ok=True)
            t = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))
            path = f'../model/last_SDNet_{t}.pth'
            train_pipeline.model.eval()
            torch.save(train_pipeline.model.state_dict(), path)
            print(f'🎉 训练结束，已将当前模型保存到 {os.path.join(os.getcwd(), path)}')
            exit()
    return wrapper


class TrainPipeline:
    """ 训练流水线 """

    def __init__(self, train_dataset_dir: str, test_dataset_dir: str, lr=0.01, step_size=15,
                 train_batch_size=10, test_batch_size=10, epochs=20, test_freq=5, use_gpu=True, log_dir=None):
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

        log_dir: str
            记录训练日志的文件夹路径
        """
        self.lr = lr
        self.epochs = epochs
        self.test_freq = test_freq
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.model = SDNet().to(self.device)
        self.logger = SummaryWriter(log_dir)
        # 创建数据集和数据加载器
        self.train_dataset = SDDataset(train_dataset_dir)
        self.test_dataset = SDDataset(test_dataset_dir)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=train_batch_size, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=test_batch_size, shuffle=True)
        # 定义优化器和损失函数
        self.criterion = BCELoss()
        self.optimizer = Adam(self.model.parameters(), lr=0.01)
        self.lr_scheduler = StepLR(
            optimizer=self.optimizer, step_size=step_size, gamma=0.1)

    @save_model
    def train(self):
        """ 训练模型 """
        train_losses = []
        test_losses = []
        bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
        print('🚀 开始训练！')
        for e in range(self.epochs):
            with tqdm(self.train_loader, bar_format=bar_format) as train_bar:
                train_bar.set_description(f"\33[36m🌌 Epoch {e + 1:3d} ")
                start_time = datetime.now()
                self.model.train()
                for images, masks in self.train_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    masks_hat = self.model(images)
                    self.optimizer.zero_grad()
                    train_loss = self.criterion(masks_hat, masks)
                    train_loss.backward()
                    self.optimizer.step()
                    cost_time = datetime.now() - start_time
                    train_bar.set_postfix_str(
                        f'train_loss={train_loss.item():.5f}，执行时间：{cost_time}\33[0m')
                    train_bar.update()

            # 测试模型
            if (e+1) % self.test_freq == 0:
                with tqdm(self.test_loader, bar_format=bar_format) as test_bar:
                    test_bar.set_description('\33[35m🛸 测试中')
                    start_time = datetime.now()
                    self.model.eval()
                    for images, masks in self.test_loader:
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                        masks_hat = self.model(images)
                        test_loss = self.criterion(masks_hat, masks)
                        test_bar.set_postfix_str(
                            f'test_loss={test_loss.item():.5f}，执行时间：{cost_time}\33[0m')
                        test_bar.update()

            # 记录误差
            train_loss, test_loss = train_loss.item(), test_loss.item()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            self.logger.add_scalars(
                "loss curve", {"train loss": train_loss, "test loss": test_loss}, e)
            self.lr_scheduler.step()

        self.logger.close()
        return train_losses, test_losses


if __name__ == '__main__':
    train_config = {
        'lr': 0.1,
        'epochs': 20,
        'test_freq': 1,
        'train_dataset_dir': '../data/specular-dataset/Train',
        'test_dataset_dir': '../data/specular-dataset/Test',
        'train_batch_size': 12,
        'test_batch_size': 12,
        'log_dir': '../log',
        'use_gpu': True,
    }
    train_pipeline = TrainPipeline(**train_config)
    train_losses, test_losses = train_pipeline.train()

    plt.style.use(['matlab'])
    plt.plot(range(1, train_config["epochs"]+1), train_losses)
    plt.plot(range(1, train_config["epochs"]+1), test_losses)
    plt.title('Loss curve')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'test loss'])
    plt.show()
