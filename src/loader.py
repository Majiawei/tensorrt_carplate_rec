# coding=utf-8

# imports
import os

import cv2
import numpy
import torch
from torch.utils.data import Dataset, DataLoader


# -------    loader section(custom needed)    -------
class MyDataset(Dataset):
    def __init__(self, label_path, alphabet, resize,
                 img_root=''):
        super(MyDataset, self).__init__()
        self.img_root = img_root
        self.labels = self.get_labels(label_path)
        self.alphabet = alphabet
        self.width, self.height = resize

    def __getitem__(self, index):
        image_name = list(self.labels[index].keys())[0]
        label = list(self.labels[index].values())[0]
        path = os.path.join(self.img_root, image_name)

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        image = cv2.resize(image, (0, 0), fx=self.width / w, fy=self.height / h, interpolation=cv2.INTER_CUBIC)
        image = (numpy.reshape(image, (self.height, self.width, 1))).transpose(2, 0, 1)
        image = self.preprocessing(image)
        return image, index, label

    def __len__(self):
        return len(self.labels)

    def get_labels(self, label_path):
        # return text labels in a list
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = [{c.strip().split(' ')[0]: c.strip().split(' ')[1]} for c in file.readlines()]

        return labels

    def preprocessing(self, image):
        ## already have been computed
        mean = 0.588
        std = 0.193
        image = image.astype(numpy.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(mean).div_(std)

        return image


# 加载数据集需要用到的参数
label_txt = '/data/zhiwei.dong/datasets/quantize/quantize_data.txt'
# label_txt = '/home/yzzc/Work/lq/ezai/all_projects/carplate_recognition/data/test_new/split.txt'
img_H = 32
img_W = 100
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新领学使警挂港澳电"
batch_size = 1
test_data = MyDataset(label_path=label_txt, alphabet=alphabet, resize=(img_W, img_H))
data_loader = DataLoader(dataset=test_data, batch_size=batch_size)
