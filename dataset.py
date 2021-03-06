import rawpy
import numpy as np
import random
import torch as meg
from torch.utils.data import Dataset


def imageCrop(rggb, size):
    h, w = size
    overall_h, overall_w = rggb.shape[1], rggb.shape[2]
    if h >= overall_h and w >= overall_w:
        padding_x = h % overall_h
        padding_y = w % overall_w
        rggb = np.pad(rggb, ((0, 0), (padding_x // 2, padding_x - padding_x // 2),
                             (padding_y // 2, padding_y - padding_y // 2)), 'reflect')
        return np.array([rggb])

    # first padding
    rggb = np.pad(rggb, ((0, 0), (8, 8), (8, 8)), 'reflect')
    rh, rw = h + 16, w + 16
    img_list = list()
    rows = overall_h // h
    cols = overall_w // w
    for row in range(rows):
        for col in range(cols):
            img_list.append(rggb[:, h * row:h * row + rh, w * col:w * col + rw])
        if overall_w % w:
            seg = rggb[:, h * row:h * row + rh, w * cols:]
            temp = np.pad(seg, ((0, 0), (0, 0),
                                (0, rw-seg.shape[2])), 'reflect')
            assert temp.shape[1:] == (rh, rw)
            img_list.append(temp)
    if overall_h % h:
        for col in range(cols):
            seg = rggb[:, h * rows:, w * col:w * col + rw]
            temp = np.pad(seg, ((0, 0), (0, rh-seg.shape[1]), (0, 0)),
                          'reflect')
            assert temp.shape[1:] == (rh, rw)
            img_list.append(temp)
        if overall_w % w:
            seg = rggb[:, h * rows:, w * cols:]
            temp = np.pad(seg, ((0, 0), (0, rh-seg.shape[1]), (0, rw-seg.shape[2])),
                          'reflect')
            assert temp.shape[1:] == (rh, rw)
            img_list.append(temp)
    return np.array(img_list).astype(np.float32)


class G_Exchange(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, inputs):
        sample, sample_gt = inputs
        rand = random.random()
        if rand > self.prob:
            sample[1, :, :], sample[2, :, :] = sample[2, :, :], sample[1, :, :]
            if sample_gt is not None:
                sample_gt[1, :, :], sample_gt[2, :, :] = sample_gt[2, :, :], sample_gt[1, :, :]
        return sample, sample_gt


class BrightnessContrast(object):
    def __init__(self, norm_num, prob=0.5):
        self.prob = prob
        self.norm_num = norm_num

    def __call__(self, inputs):
        sample, sample_gt = inputs
        h, w = sample.shape[1:]
        if meg.rand(1) < self.prob:
            alpha = meg.rand(1) + 0.5
            beta = (random.random() * 150 + 50) / self.norm_num
            bbeta = meg.full((4, h, w), beta)
            sample = alpha * sample + bbeta
            if sample_gt is not None:
                sample_gt = alpha * sample_gt + bbeta
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
        sample = meg.from_numpy(self.data[item])
        if self.isTrain:
            sample_gt = meg.from_numpy(self.gt[item])
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
    a = np.pad(a, ((0,0), (0,0), (1,2)), 'reflect')
    print(a.shape)
