import cv2
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader
import torch as meg


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


def show(img):
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('result_image.jpg', img)


def pack_raw(raw):
    postprocess = raw.raw_image_visible.astype(np.float32)
    # raw数据归一化处理，white_level: sensor的白电平，black_level: sensor的黑电平
    white_level = np.max(raw.raw_image)
    black_level = raw.black_level_per_channel[0]
    rggb = np.maximum(postprocess - black_level, 0) / \
          (white_level - black_level)
    R = rggb[0::2, 0::2]  # [0,0]
    Gr = rggb[0::2, 1::2]  # [0,1]
    Gb = rggb[1::2, 0::2]  # [1,0]
    B = rggb[1::2, 1::2]  # [1,1]
    out = np.stack((R, Gr, Gb, B))
    raw.close()
    return out


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


class RawUtils:

    @classmethod
    def bggr2rggb(cls, *bayers):
        res = []
        for bayer in bayers:
            res.append(bayer[::-1, ::-1])
        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def rggb2bggr(cls, *bayers):
        return cls.bggr2rggb(*bayers)

    @classmethod
    def bayer2rggb(cls, *bayers):
        res = []
        for bayer in bayers:
            H, W = bayer.shape
            res.append(
                bayer.reshape(H//2, 2, W//2, 2)
                .transpose(0, 2, 1, 3)
                .reshape(H//2, W//2, 4)
            )
        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def rggb2bayer(cls, *rggbs):
        res = []
        for rggb in rggbs:
            H, W, _ = rggb.shape
            res.append(
                rggb.reshape(H, W, 2, 2)
                .transpose(0, 2, 1, 3)
                .reshape(H*2, W*2)
            )

        if len(res) == 1:
            return res[0]
        return res

    @classmethod
    def bayer2rgb(cls, *bayer_01s, wb_gain, CCM, gamma=2.2):

        wb_gain = np.array(wb_gain)[[0, 1, 1, 2]]
        res = []
        for bayer_01 in bayer_01s:
            bayer = cls.rggb2bayer(
                (cls.bayer2rggb(bayer_01) * wb_gain).clip(0, 1)
            ).astype(np.float32)
            bayer = np.round(np.ascontiguousarray(bayer) * 65535).clip(0, 65535).astype(np.uint16)
            rgb = cv2.cvtColor(bayer, cv2.COLOR_BAYER_BG2RGB_EA).astype(np.float32) / 65535
            rgb = rgb.dot(np.array(CCM).T).clip(0, 1)
            rgb = rgb ** (1/gamma)
            res.append(rgb.astype(np.float32))

        if len(res) == 1:
            return res[0]
        return res
