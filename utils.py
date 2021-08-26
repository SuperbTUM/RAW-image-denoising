import cv2
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch as meg
import rawpy
import matplotlib.pyplot as plt
from load_data import pack_raw, unpack


def rgb2gray(rgbs):
    assert rgbs.shape[-1] == 3
    return 0.2126 * rgbs[:, :, :, 0] + 0.7152 * rgbs[:, :, :, 1] + 0.0722 * rgbs[:, :, :, 2]


def drawLossCurve(loss_mean):
    assert len(loss_mean) > 0
    # plt.figure()
    plt.plot(loss_mean, linewidth=2)
    plt.xlabel('epoch')
    plt.ylabel('MSE loss (sum/mean)')
    plt.title('training loss curve')
    # plt.show()
    plt.savefig("loss_curve.png")
    return


def saveCheckpoint(model, l0loss, optimizer, lr, path):
    model.eval()
    meg.save({
        'loss': l0loss,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr': lr
    }, path
    )


def loadCheckpoint(model, optimizer, path, cuda):
    if cuda:
        checkpoint = meg.load(path)
    else:
        checkpoint = meg.load(path, map_location=meg.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    l0loss = checkpoint['loss']
    lr = checkpoint['lr']
    model.eval()
    backup_info = (model, optimizer, l0loss, lr)
    return backup_info


def show_and_save(img, raw):
    img = img.transpose(1, 2, 0)
    new_raw = unpack(img)
    raw.raw_image_visible[:] = new_raw
    rgb_img = raw.postprocess(use_camera_wb=True)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', rgb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('result_image.jpg', rgb_img)


def norm(img):
    norm_num = np.max(img)
    img /= norm_num
    return img, norm_num


def L0loss(predict, gt, gamma=0.5, reduction='mean'):
    loss = (meg.abs(predict - gt) + 1e-8) ** gamma
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction is None:
        return loss
    else:
        raise NotImplementedError


def amendment(rggb):
    rggb[:, 0, 0] = (rggb[:, 1, 0] + rggb[:, 1, 1] + rggb[:, 0, 1]) / 3  # top-left
    rggb[:, -1, 0] = (rggb[:, -2, 0] + rggb[:, -1, 1] + rggb[:, -2, 1]) / 3  # bottom-left
    rggb[:, 0, -1] = (rggb[:, 1, -1] + rggb[:, 0, -2] + rggb[:, 1, -2]) / 3  # top-right
    rggb[:, -1, -1] = (rggb[:, -1, -2] + rggb[:, -2, -1] + rggb[:, -2, -2]) / 3  # bottom-right
    return rggb


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    raw = rawpy.imread('img_data/test.ARW')
    rggb = pack_raw(raw)  # rggb data
    rggb += 200
    new_raw = unpack(rggb)  # bayer data (H, W)
    raw.raw_image_visible[:] = new_raw
    rgb = raw.postprocess(use_camera_wb=True)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('test_img.jpg', rgb)
