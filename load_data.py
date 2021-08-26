import numpy as np
from dataset import imageCrop
import random
import rawpy


def getRawInfo():
    info = {
        "Exposure Time": 0.25,
        "Aperture": "f 1/16",
        'ISO': 12800,
        "Exposure Compensation": 0
    }
    return info


def pack_raw(vision):
    res = []
    H, W = vision.shape
    res.extend(
        vision.reshape(H // 2, 2, W // 2, 2)
            .transpose(0, 2, 1, 3)
            .reshape(H // 2, W // 2, 4)
    )
    return np.array(res)


def unpack(rggb):
    res = []
    H, W, _ = rggb.shape
    res.extend(
        rggb.reshape(H, W, 2, 2)
            .transpose(0, 2, 1, 3)
            .reshape(H * 2, W * 2)
    )
    return np.array(res)


def loadTestData(path, size):
    raw = rawpy.imread(path)
    visible = raw.raw_image_visible.astype(np.float32)
    rggb = pack_raw(visible)
    rggb = rggb.transpose(2, 0, 1)
    data = imageCrop(rggb, size=size)
    return data, raw, rggb.shape


def loadPairedData(paths, size):
    unpro, gt = paths
    unpro_raw = rawpy.imread(unpro)
    gt_raw = rawpy.imread(gt)

    unpro = unpro_raw.raw_image_visible[:].astype(np.float32)
    gt = gt_raw.raw_image_visible[:].astype(np.float32)

    unpro, gt = HorizontalFlip()((unpro, gt))
    unpro, gt = VerticalFlip()((unpro, gt))

    train_rggb = pack_raw(unpro).transpose(2, 0, 1)
    gt_rggb = pack_raw(gt).transpose(2, 0, 1)

    train_data = imageCrop(train_rggb, size)
    gt_data = imageCrop(gt_rggb, size)
    return train_data, gt_data, unpro_raw, gt_raw, train_rggb.shape


class HorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, raw_inputs):
        unpro, gt = raw_inputs
        if random.random() > self.prob:
            unpro = np.fliplr(unpro)
            unpro = unpro[1:-1]
            unpro = np.pad(unpro, (1, 1), "reflect")

            gt = np.fliplr(gt)
            gt = gt[1:-1]
            gt = np.pad(gt, (1, 1), "reflect")

        return unpro.copy(), gt.copy()


class VerticalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, raw_inputs):
        unpro, gt = raw_inputs
        if random.random() > self.prob:
            unpro = np.flipud(unpro)
            unpro = unpro[1:-1]
            unpro = np.pad(unpro, (1, 1), "reflect")

            gt = np.flipud(gt)
            gt = gt[1:-1]
            gt = np.pad(gt, (1, 1), "reflect")

        return unpro.copy(), gt.copy()
