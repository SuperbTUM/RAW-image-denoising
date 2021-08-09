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


def show_and_save(img):
    img = np.uint8(img)
    g = 0.5 * (img[:, :, 1] + img[:, :, 2])
    rgb_img = np.stack((img[:, :, 0], g, img[:, :, 3]), axis=-1) * 255
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', rgb_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('result_image.jpg', rgb_img)


def pack_raw(raw):
    res = []
    vision = raw.raw_image_visible.astype(np.float32)
    H, W = vision.shape
    res.extend(
        vision.reshape(H // 2, 2, W // 2, 2)
            .transpose(0, 2, 1, 3)
            .reshape(H // 2, W // 2, 4)
    )
    raw.close()
    return np.array(res)


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


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


if __name__ == '__main__':
    raw = rawpy.imread('img_data/train.ARW')
    rggb = pack_raw(raw)
    # from PMRID
    rggb = rggb.clip(0, 1)
    H, W = rggb.shape[:2]
    ph, pw = (32 - (H % 32)) // 2, (32 - (W % 32)) // 2
    rggb = np.pad(rggb, [(ph, ph), (pw, pw), (0, 0)], 'constant')
    inp_rggb = rggb.transpose(2, 0, 1)[np.newaxis]
    print(inp_rggb)
