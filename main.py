from dataset import load_image, imageCrop, NewDataset, UpsideDown, collate
from load_model import settings
from predict import prediction, recovery
from utils import *

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

import gc


def PSNR(predict, gt):
    diff = predict - gt
    diff = diff.flatten()
    rmse = meg.sqrt((diff ** 2).mean())
    return 20 * meg.log10(1. / rmse)


def test(model, val_data, batch_size):
    val_dataset = DataLoaderX(val_data, batch_size=batch_size, collate_fn=collate, num_workers=0)
    iterator = tqdm(val_dataset)
    cnt = 0
    psnr = 0.
    for sample in iterator:
        cnt += 1
        out = model(sample['data'])
        psnr += PSNR(out, sample['gt'])
    iterator.set_description('PSNR is {:.1f}'.format(psnr / cnt))
    return psnr / cnt


if __name__ == '__main__':
    size = (80, 80)
    model, optimizer, lr_scheduler = settings()

    train_path = 'img_data/train.ARW'
    train_raw = load_image(train_path)
    rggb_train = pack_raw(train_raw)
    rggb_train = rggb_train.transpose(2, 0, 1)
    rggb_train, _ = norm(rggb_train)

    train_data = imageCrop(rggb_train, size=size)

    gt_path = 'img_data/groundtruth.ARW'
    gt_raw = load_image(gt_path)
    rggb_gt = pack_raw(gt_raw)
    rggb_gt = rggb_gt.transpose(2, 0, 1)
    rggb_gt, _ = norm(rggb_gt)
    gt_data = imageCrop(rggb_gt, size=size)

    max_epoch, batch_size = 5, 10
    cur_epoch = 0
    psnr_best = 0.

    transform = Compose(
        [UpsideDown()]
    )
    assert train_data.shape == gt_data.shape
    train_dataset = NewDataset(train_data, gt_data, transform=transform)
    val_dataset = NewDataset(train_data, gt_data, isTrain=0, transform=transform)
    print("\n------------------------Start training----------------------------------")
    while True:
        if cur_epoch > 0 and cur_epoch % 2 == 0:
            print('Test phase.\n')
            model.eval()
            val_dataset.set_mode('test')
            psnr = test(model, val_dataset, batch_size)
            model = model.train()
            psnr_best = max(psnr, psnr_best)
            print('Cur psnr:{:1f} dB\tBest psnr:{:1f} dB'.format(psnr, psnr_best))

        train_dataset.set_mode('train')
        # train_dataset = DataLoader(train_data, sampler=RandomSampler(train_data, batch_size=batch_size),
        #                            num_workers=1)
        train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                                       num_workers=1)
        iterator = tqdm(train_dataloader)
        for sample in iterator:
            optimizer.zero_grad()
            predict = model(sample['data'])
            true_data = sample['gt']
            loss = M.MSELoss()(predict.view(predict.shape[0], -1), true_data.view(true_data.shape[0], -1))
            loss.backward()
            status = "epoch:{}, lr:{:2e}, loss:{:2e}".format(cur_epoch,
                                                             lr_scheduler.get_lr()[0], loss)
            iterator.set_description(status)
            # meg.optimizer.clip_grad_norm(model.parameters(), 10.0)
            M.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
        cur_epoch += 1
        if cur_epoch >= max_epoch:
            break
        gc.collect()
    saveCheckpoint(model, cur_epoch, optimizer, loss, lr_scheduler.get_lr()[0], 'checkpoint.pth')
    train_raw.close()
    gt_raw.close()
    gc.collect()
    print("----------------------------Training completed-------------------------")
    print("----------------------------Start prediction---------------------------")

    # model, optimizer, epoch, loss, lr = loadCheckpoint(model, optimizer, 'checkpoint.pth')
    predict_path = 'img_data/test.ARW'
    predict_raw = load_image(predict_path)
    rggb_predict = pack_raw(predict_raw)
    rggb_predict = rggb_predict.transpose(2, 0, 1)
    rggb_predict, norm_num = norm(rggb_predict)

    ori_shape = rggb_predict.shape
    predict_data = imageCrop(rggb_predict, size=size)
    predict_dataset = NewDataset(predict_data, isTrain=-1)

    output = prediction(predict_dataset, model)
    print("---------------------------Display results----------------------------")
    rggb_img = recovery(ori_shape, output, size)
    show_and_save(rggb_img, norm_num, predict_raw)
    predict_raw.close()

