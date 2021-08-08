import rawpy
import numpy as np
from math import *
import torch as meg
from torch.utils.data import Dataset


def load_image(path):
    raw = rawpy.imread(path)
    # raw.close()
    return raw


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
        if sample_gt is None:
            return sample, None if np.random.random(1) < self.prob else sample[::-1], None
        if np.random.random(1) < self.prob:
            return sample, sample_gt
        return sample[::-1], sample_gt[::-1]


def collate(batch):
    data = list()
    gt = list()
    for sample in batch:
        data.append(meg.from_numpy(sample['data']))
        gt.append(sample['gt'])
    data = meg.stack(data)
    gt = meg.stack(gt)
    return {'data': data, 'gt': gt}


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






