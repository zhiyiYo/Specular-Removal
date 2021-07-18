import torch
import numpy as np
# coding: utf-8
import os
from random import choice

import torchvision.transforms as T
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from specular_removal import SRNet

model = SRNet().to('cuda:0')
model.load_state_dict(torch.load('model/SRNet_2021-07-18_23-22-19.pth'))
model.eval()

# 随机选取一张图片
# image_path = 'data/SHIQ_data/test/14001_A.png'
dir_path = 'data/specular-dataset/Test/images'
image_path = choice(os.listdir(dir_path))
image_path = os.path.join(dir_path, image_path)
I = Image.open(image_path)

# 预测结果
M, S, D = model.predict(I)

# 显示图像
mpl.rc_file('resource/style/image_process.mplstyle')
fig, axes = plt.subplots(1, 4, num='高光检测')
titles = ['Original image', 'Specular mask',
          'Specular image', 'Specular removed image']
images = [I, M, S, D]
for i, (ax, image, title) in enumerate(zip(axes, images, titles)):
    cmap = plt.cm.gray if title == 'Specular mask' else None
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
plt.show()
