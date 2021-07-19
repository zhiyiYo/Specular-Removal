# coding:utf-8
import torch
from torch.utils.tensorboard import SummaryWriter

from specular_removal import SRNet


net = SRNet()
with SummaryWriter('log', comment='高光去除模型') as w:
    w.add_graph(net, torch.zeros(1, 3, 224, 224))
