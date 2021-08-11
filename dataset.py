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


class UpsideDown(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if sample_gt is None:
            return (sample, None) if np.random.random(1) < self.prob else (sample.flip(dims=[1,2]), None)
        if np.random.random(1) < self.prob:
            return sample, sample_gt
        return sample.flip(dims=[1,2]), sample_gt.flip(dims=[1,2])


class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if sample_gt is None:
            return (sample, None) if np.random.random(1) < self.prob else (sample.flip(dims=[2]), None)
        if np.random.random(1) < self.prob:
            return sample, sample_gt
        return sample.flip(dims=[2]), sample_gt.flip(dims=[2])


class VerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if sample_gt is None:
            return (sample, None) if np.random.random(1) < self.prob else (sample.flip(dims=[1]), None)
        if np.random.random(1) < self.prob:
            return sample, sample_gt
        return sample.flip(dims=[1]), sample_gt.flip(dims=[1])


class BrightnessContrast(object):
    def __init__(self, norm_num, prob=0.5):
        self.prob = prob
        self.norm_num = norm_num

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if meg.rand(1) < self.prob:
            alpha = meg.rand(1) + 0.5
            beta_ = meg.rand(1) * 150 + 50
            beta = meg.Tensor([beta_ / self.norm_num[i] for i in range(4)])
            sample = sample.permute(1, 2, 0)
            sample = alpha * sample + beta
            sample = sample.permute(2, 0, 1)
            if sample_gt is not None:
                sample_gt = sample_gt.permute(1, 2, 0)
                sample_gt = alpha * sample_gt + beta
                sample_gt = sample_gt.permute(2, 0, 1)
        return sample, sample_gt


def collate(batch):
    data = list()
    gt = list()
    for sample in batch:
        data.append(sample['data'])
        if sample['gt'] is not None:
            gt.append(sample['gt'])
    data = meg.stack(data)
    if gt:
        gt = meg.stack(gt)
    return {'data': data, 'gt': gt}


class NewDataset(Dataset):
    def __init__(self, dataset, ground_truth=None, transform=None, isTrain=1):
        super().__init__()
        self.transform = transform
        self.isTrain = isTrain
        train_size = floor(0.8 * len(dataset))
        if self.isTrain == 1:
            self.data = dataset[:train_size]
            self.gt = ground_truth[:train_size]
        elif self.isTrain == 0:
            self.data = dataset[train_size:]
            self.gt = ground_truth[train_size:]
        else:
            self.data = dataset
            self.gt = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.isTrain >= 0:
            sample_gt = self.gt[item]
            assert sample.shape == sample_gt.shape
            if self.transform:
                sample, sample_gt = self.transform((sample, sample_gt))
            return {'data': sample, 'gt': sample_gt}
        else:
            if self.transform:
                sample, _ = self.transform((sample, None))
            return {'data': sample, 'gt': None}

    def set_mode(self, mode='None'):
        self.mode = mode


if __name__ == "__main__":
    a = np.ones((4, 10, 10))
    a = meg.from_numpy(a)
    a = a.permute(1, 2, 0)
    print(a.shape)
    b = np.array([1,1,1,1])
    1.5 * a + b




