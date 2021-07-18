# coding:utf-8
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class SpecularAreaSelector:
    """高光区域选择器"""

    def __init__(self, image: np.ndarray):
        """
        Parameters
        ----------
        image: ~np.ndarray
            BGR 图像
        """
        self.image = image
        self.msf_image = None
        self.chroma_image = None
        self.specular_mask = None

    def __getMSFImage(self) -> np.ndarray:
        """获取 MSF 图像"""
        V_min = np.min(self.image, axis=2, keepdims=True)  # type:np.ndarray
        self.mean_V_mean = np.sum(V_min) / (V_min.shape[0] * V_min.shape[1])
        self.msf_image = np.uint8(self.image - V_min + self.mean_V_mean)
        self.chroma_image = self.msf_image / \
            self.msf_image.sum(axis=2, keepdims=True)
        return self.msf_image

    def select(self) -> np.ndarray:
        """选择高光区域"""
        self.__getMSFImage()
        mask = (self.image - self.msf_image) > self.mean_V_mean
        self.specular_mask = np.uint8(np.all(mask, axis=2) * 255)
        specular_area = cv.bitwise_and(
            self.image, self.image, mask=self.specular_mask)
        return specular_area


if __name__ == "__main__":
    # type:np.ndarray
    image = cv.imread("../data/specular-dataset/Train/images/00027.png")
    selector = SpecularAreaSelector(image)
    specular_area = selector.select()

    plt.style.use(["image_process"])
    fig, axes = plt.subplots(
        1, 3, num="镜面反射选择", tight_layout=True, sharey=True)
    titles = ["Original image", "Specular area", "MSF image"]
    images = [image, specular_area, selector.msf_image]
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
        ax.set_title(titles[i])
    plt.show()
