import cv2
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch as meg
import rawpy
import matplotlib.pyplot as plt
from dataset import load_image, imageCrop
from K_Sigma_transform import KSigma
from scipy.optimize import leastsq


def getRawInfo():
    info = {
        "Exposure Time": 0.25,
        "Aperture": "f 1/16",
        'ISO': 12800,
        "Exposure Compensation": 0
    }
    return info


def rgb2gray(rgbs):
    assert rgbs.shape[-1] == 3
    return 0.2126 * rgbs[:, :, :, 0] + 0.7152 * rgbs[:, :, :, 1] + 0.0722 * rgbs[:, :, :, 2]


def cal_kb(rgbs):
    def fun(p, x):
        k, b = p
        return k * x + b

    def error(p, x, y):
        return fun(p, x) - y
    grayscales = rgb2gray(rgbs)
    mean = grayscales.mean(dim=1)
    var = grayscales.var(dim=1, unbiased=True)
    mean = mean.flatten().numpy()
    var = var.flatten().numpy()
    p0 = np.array([1, 3])
    param = leastsq(error, p0, args=(mean, var))
    k, b = param[0]
    return k, b


def ksigmaTransform(rggb, V=65024, inverse=False):
    K_coeff = (0.0005995267, 0.00868861)
    B_coeff = (7.11772e-7, 6.514934e-4, 0.11492713)
    anchor = 1600
    ksigma = KSigma(K_coeff, B_coeff, anchor, V)
    return ksigma(rggb, getRawInfo()['ISO'], inverse=inverse)


def loadTrainableData(path, size):
    raw = load_image(path)
    rggb = pack_raw(raw)
    rggb = rggb.transpose(2, 0, 1)
    data = imageCrop(rggb, size=size)
    return data, raw, rggb.shape


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


def loadCheckpoint(model, optimizer, path):
    checkpoint = meg.load(path)
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
