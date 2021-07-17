# coding:utf-8
import os

from PIL import Image

dir_path = 'resource/images'
file_paths = [os.path.join(dir_path, i) for i in os.listdir(dir_path)]
for file_path in file_paths:
    path, suffix = os.path.splitext(file_path)
    if suffix.lower() != '.jpg':
        image = Image.open(file_path).convert('RGB')
        image.save(path + '.jpg')
