from dataset import load_image, imageCrop, pack_raw, unpack, NewDataset, UpsideDown
from load_model import settings
from predict import prediction, recovery
from utils import show

import rawpy
# import megengine as meg
# import megengine.module as M
# import megengine.functional as F
import numpy as np
# from megengine.data.dataset import Dataset
from math import *
from tqdm import *
# from megengine.data import DataLoader, RandomSampler
# from megengine.data.transform import Compose

import torch as meg
import torch.nn as M
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from torch import optim
from torchsummary import summary
from torch.autograd import Variable


def PSNR(predict, gt):
    diff = predict - gt
    diff = diff.flatten()
    rmse = meg.sqrt((diff ** 2).mean())
    return 20 * meg.log10(1. / rmse)


def test(model, val_data, batch_size):
    val_dataset = DataLoader(val_data, batch_size=batch_size, num_workers=1)
    iterator = tqdm(val_dataset)
    cnt = 0
    psnr = 0.
    for sample in iterator:
        cnt += 1
        out = model(sample['data'])
        psnr += PSNR(out, sample['gt'])
    iterator.set_description('PSNR is {.1f}'.format(psnr / cnt))
    return psnr / cnt


if __name__ == '__main__':
    train_path = 'img_data/train.ARW'
    train_raw = load_image(train_path)
    rggb_train, _, _ = pack_raw(train_raw)
    size = (24, 32)
    train_data = imageCrop(rggb_train, size=size)
    gt_path = 'img_data/groundtruth.ARW'
    gt_raw = load_image(gt_path)
    rggb_gt, _, _ = pack_raw(gt_raw)
    gt_data = imageCrop(rggb_gt, size=size)
    model, optimizer, lr_scheduler = settings()
    # print(summary(model))
    max_epoch, batch_size = 5, 20
    cur_epoch = 0
    psnr_best = 0.
    train_data = NewDataset(train_data, gt_data, transform=Compose([UpsideDown()]))
    val_data = NewDataset(train_data, gt_data, isTrain=False, transform=Compose([UpsideDown()]))
    while True:
        if cur_epoch > 0 and cur_epoch % 2 == 0:
            print('Test phase.\n')
            model.eval()
            val_data.set_mode('test')
            psnr = test(model, val_data, batch_size)
            model = model.train()
            psnr_best = max(psnr, psnr_best)
            print('Cur psnr:{}\tBest psnr:{}'.format(psnr, psnr_best))

        train_data.set_mode('train')
        # train_dataset = DataLoader(train_data, sampler=RandomSampler(train_data, batch_size=batch_size),
        #                            num_workers=1)
        train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=1)
        iterator = tqdm(train_dataset)
        for sample in iterator:
            optimizer.zero_grad()
            predict = Variable(model(sample['data']))
            true_data = Variable(sample['gt'])
            loss = M.MSELoss(predict.view(batch_size, -1), true_data.view(batch_size, -1))
            loss.backward()
            # meg.optimizer.clip_grad_norm(model.parameters(), 10.0)
            M.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
        cur_epoch += 1
        if cur_epoch >= max_epoch:
            break
    train_raw.close()
    gt_raw.close()

    predict_path = 'img_data/test.ARW'
    predict_raw = load_image(predict_path)
    rggb_predict, black_level, white_level = pack_raw(predict_raw)
    ori_shape = rggb_predict.shape
    predict_data = imageCrop(rggb_predict, size=size)
    predict_dataset = NewDataset(predict_data)

    output = prediction(predict_dataset, model)

    rggb_img = recovery(ori_shape, output, size)
    img = unpack(rggb_img, black_level, white_level)
    visualize = img.postprocess()
    show(visualize)

    predict_raw.close()
