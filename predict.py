import numpy as np
from tqdm import *
from utils import DataLoaderX
from dataset import collate
from math import *
from utils import amendment


def prediction(data, model):
    data_loader = DataLoaderX(data, batch_size=20, collate_fn=collate, num_workers=0)
    model.training = False
    iterator = tqdm(data_loader)
    out = []
    for sample in iterator:
        out += model(sample['data'])
    return out


def recovery(ori_shape, output, size):
    cols = ceil(ori_shape[2] / size[1])
    rows = ceil(ori_shape[1] / size[0])
    assert rows * cols == len(output)
    results = np.zeros((ori_shape[0], rows*size[0], cols*size[1]))
    for i, out in enumerate(output):
        out = out.detach().numpy()
        out = amendment(out)
        end_col = (i + 1) % cols * size[1] if (i+1) % cols > 0 else cols*size[1]
        results[:, i // cols * size[0]:(i // cols + 1) * size[0],
        i % cols * size[1]:end_col] = out
    return results[:, 0:ori_shape[1], 0:ori_shape[2]]
