# coding:utf-8
import os
import cv2 as cv
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset


class SDDataset(Dataset):
    """ 反光检测数据集 """

    def __init__(self, dir_path: str):
        """
        Parameters
        ----------
        dir_path: str
            数据文件夹路径，要求目录下包含以下文件夹：
            * images
            * masks
        """
        super().__init__()
        self.image_dir_path = os.path.join(dir_path, 'images')
        self.mask_dir_path = os.path.join(dir_path, 'masks')
        self.image_paths = [os.path.join(self.image_dir_path, i)
                            for i in os.listdir(self.mask_dir_path)]
        self.mask_paths = [os.path.join(self.mask_dir_path, i)
                           for i in os.listdir(self.mask_dir_path)]
        # 检查图像数量是否对应
        self.__checkDataset()

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        transform = T.Compose([T.ToTensor(), T.Resize(512), T.CenterCrop(512)])
        image = transform(cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB))
        mask = transform(Image.open(mask_path).convert('1'))
        return image, mask

    def __len__(self):
        return len(self.image_paths)

    def __checkDataset(self):
        """ 检查数据集 """
        img_nums = [os.path.splitext(os.path.basename(i))[0]
                    for i in self.image_paths]
        mask_nums = [os.path.splitext(os.path.basename(i))[0]
                     for i in self.mask_paths]
        if img_nums != mask_nums:
            raise Exception("图像编号必须与模板编号不一致")
