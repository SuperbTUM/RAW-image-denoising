import rawpy
import numpy as np
from math import *
import torch as meg
from torch.utils.data import Dataset


def load_image(path):
    raw = rawpy.imread(path)
    # raw.close()
    return raw


def pack_raw(raw):
    postprocess = raw.raw_image_visible.astype(np.float32)
    # raw数据归一化处理，white_level: sensor的白电平，black_level: sensor的黑电平
    white_level = np.max(raw.raw_image)
    black_level = raw.black_level_per_channel[0]
    raw = np.maximum(postprocess - black_level, 0) / \
          (white_level - black_level)
    R = raw[0::2, 0::2]  # [0,0]
    Gr = raw[0::2, 1::2]  # [0,1]
    Gb = raw[1::2, 0::2]  # [1,0]
    B = raw[1::2, 1::2]  # [1,1]
    out = np.stack((R, Gr, Gb, B))
    return out, black_level, white_level


def unpack(raw, black_level, white_level):
    # 4->1 raw:(H, W, 4) --> out:(2 * H, 2 * W)
    # 创建一个和raw数据单通道大小相同的全0数组
    out = np.zeros((raw.shape[0] * 2, raw.shape[1] * 2))

    # 按R,Gr,Gb,B的顺序赋值给out数组，最后out就是raw数据的单通道数组
    out[0::2, 0::2] = raw[:, :, 0]
    out[0::2, 1::2] = raw[:, :, 1]
    out[1::2, 0::2] = raw[:, :, 2]
    out[1::2, 1::2] = raw[:, :, 3]

    # 将归一化的数据恢复到原样
    out = out * (white_level - black_level) + black_level
    out = np.minimum(np.maximum(out, black_level), white_level).astype(np.uint16)
    return out


def imageCrop(rggb, size):
    h, w = size
    overall_h, overall_w = rggb.shape[1], rggb.shape[2]
    img_list = list()
    rows = overall_h // h
    cols = overall_w // w
    for row in range(rows):
        for col in range(cols):
            img_list.append(rggb[:, h*row:h*(row+1),w*col:w*(col+1)])
        if overall_w % w:
            temp = np.zeros((rggb.shape[0], h, w))
            temp[:, 0:h, 0:(overall_w - w*cols)] = rggb[:, h*row:h*(row+1), w*cols:]
            img_list.append(temp)
    if overall_h % h:
        for col in range(cols):
            temp = np.zeros((rggb.shape[0], h, w))
            temp[:, 0:(overall_h - h*rows), 0:w] = rggb[:, h*rows:, w*col:w*(col+1)]
            img_list.append(temp)
        if overall_w % w:
            temp = np.zeros((rggb.shape[0], h, w))
            temp[:, 0:(overall_h - h*rows), 0:(overall_w - w*cols)] = rggb[:, h * rows:, w*cols:]
            img_list.append(temp)
    return meg.Tensor(img_list)
    # return np.array(img_list)


class UpsideDown:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample, sample_gt):
        if np.random.rand(1) < self.prob:
            return sample, sample_gt
        return sample[::-1], sample_gt[::-1]


class NewDataset(Dataset):
    def __init__(self, dataset, ground_truth=None, transform=None, isTrain=1):
        super().__init__()
        self.transform = transform
        train_size = floor(0.8 * len(dataset))
        if isTrain == 1:
            self.data = dataset[:train_size]
            self.gt = ground_truth[:train_size]
        elif isTrain == 0:
            self.data = dataset[train_size:]
            self.gt = ground_truth[train_size:]
        else:
            self.data = dataset.copy()
            self.gt = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        sample_gt = self.gt[item]
        if self.transform:
            sample, sample_gt = self.transform(sample, sample_gt)
        return {'data': sample, 'gt': sample_gt}

    def set_mode(self, mode='None'):
        self.mode = mode






