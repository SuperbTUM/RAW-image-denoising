import cv2
import numpy as np
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


def show(img):
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
    raw = np.maximum(postprocess - black_level, 0) / \
          (white_level - black_level)
    R = raw[0::2, 0::2]  # [0,0]
    Gr = raw[0::2, 1::2]  # [0,1]
    Gb = raw[1::2, 0::2]  # [1,0]
    B = raw[1::2, 1::2]  # [1,1]
    out = np.stack((R, Gr, Gb, B))
    return out, black_level, white_level


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