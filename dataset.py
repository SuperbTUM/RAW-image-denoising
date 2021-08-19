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
    if h >= overall_h and w >= overall_w:
        padding_x = h % overall_h
        padding_y = w % overall_w
        rggb = np.pad(rggb, ((0, 0), (padding_x // 2, padding_x - padding_x // 2),
                             (padding_y // 2, padding_y - padding_y//2)), 'constant', constant_values=0)
        return np.array([rggb])

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
    return np.array(img_list)


class UpsideDown(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        if sample_gt is None:
            return (sample, None) if np.random.random(1) < self.prob else (np.flip(sample, -1), None)
        if np.random.random(1) < self.prob:
            return sample, sample_gt
        return np.flip(sample, -1), np.flip(sample_gt, -1)


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
        h, w = sample.shape[1:]
        if meg.rand(1) < self.prob:
            alpha = np.random.rand(1) + 0.5
            beta_ = np.random.rand(1) * 150 + 50
            beta = [beta_ / self.norm_num[i] for i in range(4)]
            bbeta = np.stack([np.full((h, w), beta[i]) for i in range(4)])
            sample = alpha * sample + bbeta
            if sample_gt is not None:
                sample_gt = alpha * sample_gt + bbeta
        return sample, sample_gt


def collate(batch):
    data = list()
    gt = list()
    for sample in batch:
        data.append(meg.from_numpy(sample['data']))
        if sample['gt'] is not None:
            gt.append(meg.from_numpy(sample['gt']))
    data = meg.stack(data)
    if gt:
        gt = meg.stack(gt)
    return {'data': data, 'gt': gt}


class NewDataset(Dataset):
    def __init__(self, dataset, ground_truth=None, transform=None, isTrain=True):
        super().__init__()
        self.transform = transform
        self.isTrain = isTrain
        if self.isTrain:
            self.data = dataset
            self.gt = ground_truth
        else:
            self.data = dataset
            self.gt = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        if self.isTrain:
            sample_gt = self.gt[item]
            assert sample.shape == sample_gt.shape
            if self.transform:
                sample, sample_gt = self.transform((sample, sample_gt))
            return {'data': sample, 'gt': sample_gt}
        else:
            if self.transform:
                sample, _ = self.transform([sample, None])
            return {'data': sample, 'gt': None}

    def set_mode(self, mode='None'):
        self.mode = mode


if __name__ == "__main__":
    a = np.ones((4, 2, 2))
    b = [1,2]
    b = np.tile(b, (4, 2, 1))
    print(b)
    print(a + b)




