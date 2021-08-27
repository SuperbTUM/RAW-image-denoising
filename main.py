from dataset import *
from load_model import settings
from predict import prediction, recovery
from utils import *
from load_data import *
from K_Sigma_transform import ksigmaTransform
from tqdm import tqdm
import torch as meg
import torch.nn as M
from torchvision.transforms import Compose
from torch.autograd import Variable
import gc
from skimage.metrics import structural_similarity

# import megengine as meg
# import megengine.module as M
# import megengine.functional as F
# from megengine.data.dataset import Dataset
# from megengine.data import DataLoader, RandomSampler
# from megengine.data.transform import Compose


def PSNR(predict, gt, max_pixel=256.):
    diff = predict - gt
    diff = diff.flatten()
    rmse = meg.sqrt((diff ** 2).mean())
    return 20 * meg.log10((max_pixel-1) / rmse + 1e-8)


def test(model, val_data, batch_size, inp_scale, cuda, V, train_norm):
    val_dataset = DataLoaderX(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    iterator = tqdm(val_dataset)
    cnt = 0
    l0_loss = 0.
    for sample in iterator:
        cnt += 1
        sample['data'] = sample['data'].float()
        if cuda:
            sample['data'] = Variable(sample['data']).cuda()
        out = model(sample['data']).cpu()
        out = ksigmaTransform(out / inp_scale * train_norm, V=V, inverse=True)
        l0_loss += L0loss(out, sample['gt'])
    iterator.set_description('test loss is {:.1f}'.format(l0_loss / cnt))
    return l0_loss / cnt


if __name__ == '__main__':
    cuda = False
    size = (80, 80)
    # size = (2016, 3024)
    inp_scale = 256
    model, optimizer, lr_scheduler = settings(pretrained="torch_pretrained.ckp", cuda=cuda)
    max_epoch, batch_size = 20, 10
    cur_epoch = 0
    l0loss_least = 10.
    loss_mean = list()

    print("\n------------------------Start training----------------------------------")
    while True:
        train_data, gt_data, train_raw, gt_raw, _ = loadPairedData(('img_data/train.ARW', 'img_data/groundtruth.ARW'),
                                                                   size)
        # K-sigma transformation
        V = 2 ** 16 - train_raw.black_level_per_channel[0]
        train_data = ksigmaTransform(train_data, V=V)
        train_data, train_norm = norm(train_data)
        train_data *= inp_scale

        train_transform = Compose(
            [
                G_Exchange(),
                BrightnessContrast(train_norm)
            ]
        )
        train_dataset = NewDataset(train_data, gt_data, transform=train_transform)

        if cur_epoch > 0 and cur_epoch % 2 == 0:
            print('\nTest phase.')
            model.eval()
            train_dataset.set_mode('test')
            l0loss = test(model, train_dataset, batch_size, inp_scale, cuda=cuda, V=V, train_norm=train_norm)
            if l0loss < l0loss_least:
                l0loss_least = l0loss
                saveCheckpoint(model, l0loss, optimizer, lr_scheduler.get_last_lr()[0], 'checkpoint_cropped.pth')
            print('Cur l0loss:{:.1e}, Least l0loss:{:.1e}'.format(l0loss, l0loss_least))
            model = model.train()

        if cur_epoch >= max_epoch:
            break
        train_dataset.set_mode('train')
        train_dataloader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate,
                                       num_workers=0)
        iterator = tqdm(train_dataloader)
        loss_list = list()
        for sample in iterator:
            optimizer.zero_grad()
            if cuda:
                sample['data'] = Variable(sample['data']).cuda()
            sample['data'] = sample['data'].float()
            predict = model(sample['data']).cpu()
            predict = ksigmaTransform(predict / inp_scale * train_norm, V=V, inverse=True)
            true_data = sample['gt'].float()
            loss = M.L1Loss()(predict.view(predict.shape[0], -1),
                              true_data.view(true_data.shape[0], -1))
            # loss = L0loss(predict.view(predict.shape[0], -1), true_data.view(true_data.shape[0], -1))
            loss_list.append(loss.detach().numpy())
            loss.backward()
            status = "epoch:{}, lr:{:.2e}, loss:{:.2e}".format(cur_epoch,
                                                               lr_scheduler.get_last_lr()[0],
                                                               sum(loss_list)/len(loss_list))
            iterator.set_description(status)
            M.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            lr_scheduler.step()
        cur_epoch += 1
        loss_mean.append(sum(loss_list) / len(loss_list))
        gc.collect()

    # drawLossCurve(loss_mean)
    train_raw.close()
    gt_raw.close()
    gc.collect()
    if cuda:
        meg.cuda.empty_cache()

    print("----------------------------Training completed-------------------------")
    print("----------------------------Start prediction---------------------------")

    model, optimizer, l0loss, lr = loadCheckpoint(model, optimizer, 'checkpoint.pth', cuda)
    predict_path = 'img_data/test.ARW'
    predict_data, predict_raw, ori_shape = loadTestData(predict_path, size)
    predict_data = ksigmaTransform(predict_data, V=V)
    predict_data, norm_num = norm(predict_data)
    predict_data *= inp_scale
    predict_dataset = NewDataset(predict_data, isTrain=False)

    output = prediction(predict_dataset, model, batch_size=batch_size, cuda=cuda)
    output = ksigmaTransform(meg.stack(output) / inp_scale * norm_num, V=V, inverse=True)
    print("---------------------------Display results----------------------------")
    rggb_img = recovery(ori_shape, output, size)
    show_and_save(rggb_img, predict_raw)
    predict_raw.close()
