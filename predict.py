import numpy as np
from tqdm import *
from utils import DataLoaderX
from dataset import collate
from math import *


def prediction(data, model, batch_size, cuda):
    data_loader = DataLoaderX(data, batch_size=batch_size, collate_fn=collate, num_workers=0)
    model.training = False
    iterator = tqdm(data_loader)
    out = []
    for sample in iterator:
        sample['data'] = sample['data'].float()
        if cuda:
            out += model(sample['data']).cpu()
        else:
            out += model(sample['data'])
    return out


def recovery(ori_shape, output, size):
    if size[0] >= ori_shape[1] or size[1] >= ori_shape[2]:
        # de-padding
        output = output[0].detach().numpy()
        diff_x = size[0] - ori_shape[1]
        diff_y = size[1] - ori_shape[2]
        return output[:, diff_x // 2:-(diff_x - diff_x // 2),
                      diff_y // 2:-(diff_y - diff_y // 2)]

    h, w = size[0], size[1]
    cols = ceil(ori_shape[2] / w)
    rows = ceil(ori_shape[1] / h)
    assert rows * cols == len(output)
    results = np.zeros((ori_shape[0], rows * size[0], cols * size[1]))
    for i, out in enumerate(output):
        out = out.detach().numpy()
        out = out[:, 8:-8, 8:-8]
        end_col = (i + 1) % cols * size[1] if (i + 1) % cols > 0 else cols * size[1]
        results[:, i // cols * size[0]:(i // cols + 1) * size[0],
        i % cols * size[1]:end_col] = out
    return results[:, 0:ori_shape[1], 0:ori_shape[2]]


if __name__ == '__main__':
    a = np.zeros((4, 3, 3))
    print(a[:, 0:-1, 0:-1].shape)
