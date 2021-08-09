import cv2
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch as meg
import rawpy


def saveCheckpoint(model, epoch, optimizer, loss, lr, path):
    model.eval()
    meg.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'lr': lr
    }, path
    )


def loadCheckpoint(model, optimizer, path):
    checkpoint = meg.load(path)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    lr = checkpoint['lr']
    model.eval()
    backup_info = (model, optimizer, epoch, loss, lr)
    return backup_info


def show_and_save(img, norm_num, raw):
    img = img.transpose(1, 2, 0).clip(0, 1)
    for i in range(4):
        img[:, :, i] *= norm_num[i]
    new_raw = unpack(img)
    raw.raw_image_visible[:] = new_raw
    rgb_img = raw.postprocess(use_camera_wb=True)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', rgb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('result_image.jpg', rgb_img)


def norm(img):
    assert img.shape[0] == 4
    norm_num = []
    for i in range(img.shape[0]):
        val = max(map(max, img[i, :, :]))
        norm_num.append(val)
        img[i, :, :] /= val
    return img, norm_num


def pack_raw(raw):
    res = []
    vision = raw.raw_image_visible.astype(np.float32)
    H, W = vision.shape
    res.extend(
        vision.reshape(H // 2, 2, W // 2, 2)
            .transpose(0, 2, 1, 3)
            .reshape(H // 2, W // 2, 4)
    )
    # raw.close()
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


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    raw = rawpy.imread('img_data/groundtruth.ARW')
    rggb = pack_raw(raw)  # rggb data
    new_raw = unpack(rggb)  # bayer data (H, W)
    raw.raw_image_visible[:] = new_raw
    rgb = raw.postprocess(use_camera_wb=True)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
