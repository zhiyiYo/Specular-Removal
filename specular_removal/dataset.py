# coding:utf-8
import os

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset


class SRDataset(Dataset):
    """ 高光去除数据集 """

    def __init__(self, dir_path: str):
        """
        Parameters
        ----------
        dir_path: str
            数据文件夹路径，要求目录下包含以下 4 种类型的图片：
            * 0001_A.png，带高光图 (200 × 200)
            * 0001_D.png，去高光图
            * 0001_S.png，镜面反射分量图
            * 0001_T.png，镜面反射分量蒙版
        """
        super().__init__()
        if not os.path.exists(dir_path):
            raise Exception(f"文件夹 {dir_path} 不存在！")

        self.image_paths = [os.path.join(dir_path, i)
                            for i in os.listdir(dir_path)]
        # 检查图像数量是否正确
        self.__checkDataset()

    def __getitem__(self, index):
        transform = T.Compose([T.CenterCrop(192), T.ToTensor()])
        image_paths = self.image_paths[4*index:4*(index+1)]
        I = transform(Image.open(image_paths[0]))
        D = transform(Image.open(image_paths[1]))
        S = transform(Image.open(image_paths[2]))
        M = transform(Image.open(image_paths[3]).convert('1'))
        return I, M, S, D

    def __len__(self):
        return len(self.image_paths)//4

    def __checkDataset(self):
        """ 检查数据集 """
        if len(self.image_paths) % 4 != 0:
            raise Exception("图像数量必须是 4 的倍数")


if __name__ == '__main__':
    dataset = SRDataset('../data/SHIQ_data/train')
    data = dataset[0]
    print(data)
