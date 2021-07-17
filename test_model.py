import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image

from specular_detection.network import SDNet

model = SDNet().to('cuda:0')
#model.load_state_dict(torch.load('model/last_SDNet_2021-07-17_22-56-56.pth'))
model.eval()
image = Image.open('data/test_images_in_the_wild/053.png')
mask = model.predict(image)

# 显示图像
plt.style.use(['image_process'])
fig, axes = plt.subplots(1, 2, num='高光检测')
axes[0].imshow(image)
axes[1].imshow(mask, cmap=plt.cm.gray)

plt.show()
