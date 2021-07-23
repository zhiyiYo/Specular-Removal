# coding: utf-8
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from specular_removal import SRNet


model_path = 'model\SRNet_2021-07-23_23-00-22.pth'
model = SRNet().to('cuda:0')
model.load_state_dict(torch.load(model_path))
model.eval()

# 选取一张图片
image_path = 'resource/images/木桌.png'
I = Image.open(image_path)

# 预测结果
M, S, D = model.predict(I)

# 显示图像
mpl.rc_file('resource/style/image_process.mplstyle')
fig, axes = plt.subplots(1, 4, num='高光去除', tight_layout=True)
titles = ['Original image', 'Specular mask',
          'Specular image', 'Specular removed image']
images = [I, M, S, D]
for ax, image, title in zip(axes, images, titles):
    cmap = plt.cm.gray if title == 'Specular mask' else None
    ax.imshow(image, cmap=cmap)
    ax.set_title(title)
plt.show()
