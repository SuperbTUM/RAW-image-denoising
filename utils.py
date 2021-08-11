import cv2
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch as meg
import rawpy
import matplotlib.pyplot as plt
from dataset import load_image, imageCrop
from K_Sigma_transform import KSigma


def getRawInfo():
    info = {
        "Exposure Time": 0.25,
        "Aperture": "f 1/16",
        'ISO': 12800,
        "Exposure Compensation": 0
    }
    return info


def ksigmaTransform(rggb, inverse=False):
    K_coeff = (0.0005995267, 0.00868861)
    B_coeff = (7.11772e-7, 6.514934e-4, 0.11492713)
    anchor = 1600
    ksigma = KSigma(K_coeff, B_coeff, anchor)
    return ksigma(rggb, getRawInfo()['ISO'], inverse=inverse)


def loadTrainableData(path, size):
    raw = load_image(path)
    rggb = pack_raw(raw)
    rggb = rggb.transpose(2, 0, 1)
    rggb, norm_num = norm(rggb)
    data = imageCrop(rggb, size=size)
    return data, raw, norm_num, rggb.shape


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
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
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
    # cv2.imwrite('test_img.jpg', rgb)

