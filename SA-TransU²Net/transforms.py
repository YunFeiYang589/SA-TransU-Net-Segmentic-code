import random
from typing import List, Union

import cv2
import numpy as np
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T
from wandb.wandb_torch import torch


# src/transforms.py
class EdgeEnhance:
    def __call__(self, image, target):
        # 使用高斯模糊和锐化增强边界
        image = cv2.GaussianBlur(np.array(image.permute(1, 2, 0).numpy(), dtype=np.float32), (3, 3), 0)
        image = cv2.add(image, cv2.Laplacian(image, cv2.CV_32F, ksize=3))
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return image, target
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)  # 图像转为(C, H, W)
        target = F.to_tensor(target)  # 标签转为(H, W)

        # 关键修改：如果标签是2维，添加通道维度（变为1×H×W）
        if target.dim() == 2:
            target = target.unsqueeze(0)  # 增加一个通道维度

        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.flip_prob = prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Resize(object):
    def __init__(self, size: Union[int, List[int]], resize_mask: bool = True):
        self.size = size  # [h, w]
        self.resize_mask = resize_mask

    def __call__(self, image, target=None):
        # 确保输入是可处理的格式
        if not isinstance(self.size, (list, tuple)):
            self.size = (self.size, self.size)

        image = F.resize(image, self.size)
        if self.resize_mask is True and target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)  # 掩码使用最近邻插值

        return image, target


class RandomCrop(object):
    def __init__(self, size: int):
        self.size = size

    def pad_if_smaller(self, img, fill=0):
        # 如果图像最小边长小于给定size，则用数值fill进行padding
        min_size = min(img.shape[-2:])
        if min_size < self.size:
            ow, oh = img.size
            padh = self.size - oh if oh < self.size else 0
            padw = self.size - ow if ow < self.size else 0
            img = F.pad(img, [0, 0, padw, padh], fill=fill)
        return img

    def __call__(self, image, target):
        image = self.pad_if_smaller(image)
        target = self.pad_if_smaller(target)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target

