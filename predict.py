import numpy as np
from tqdm import *
from torch.utils.data import DataLoader


def prediction(data, model):
    data_loader = DataLoader(data, batch_size=20, num_workers=1)
    iterator = tqdm(data_loader)
    out = []
    for sample in iterator:
        out += model(sample['data'])
    return out


def recovery(ori_shape, output, size):
    cols = len(output) // size[0]
    rows = len(output) // size[1]
    assert rows * cols == len(output)
    results = np.zeros((ori_shape[0], rows, cols))
    for i, out in enumerate(output):
        results[:, i//cols*size[0]:(i//cols+1)*size[0], (i%cols)*size[1]:(i%cols+1)*size[1]] = out
    return results[:, 0:ori_shape[1], 0:ori_shape[2]]
