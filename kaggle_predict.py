from utils import *
from load_model import settings
from tqdm import tqdm
from torch.autograd import Variable
from dataset import NewDataset, collate
from predict import recovery
from skimage.metrics import peak_signal_noise_ratio
from imageio import mimsave
from load_data import loadTestData
from K_Sigma_transform import ksigmaTransform


def new_predict(train_path='kaggle_img/input_new.raw',
                gt_path='kaggle_img/gt_new.raw',
                cuda=False,
                size=(2016, 3024),
                inp_scale=256,
                batch_size=1):
    model, optimizer, _ = settings(pretrained=None, cuda=cuda)
    model, optimizer, l0loss, lr = loadCheckpoint(model, optimizer, 'checkpoint.pth', cuda)

    train_data, train_raw, ori_shape = loadTestData(train_path, size)
    gt_data, gt_raw, _ = loadTestData(gt_path, size)

    gt_rgb = gt_raw.postprocess(use_camera_wb=True)
    ori_psnr = peak_signal_noise_ratio(train_raw.postprocess(use_camera_wb=True),
                                       gt_rgb,
                                       data_range=256.)

    V = 2 ** 10 - train_raw.black_level_per_channel[0]
    train_data = ksigmaTransform(train_data, V=V)
    train_data, train_norm = norm(train_data)
    train_data *= inp_scale
    dataset = NewDataset(train_data, gt_data, transform=None)
    dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    iterator = tqdm(dataloader)
    for sample in iterator:
        sample['data'] = sample['data'].float()
        if cuda:
            sample['data'] = Variable(sample['data']).cuda()
        out = model(sample['data']).cpu()
        out = ksigmaTransform(out / inp_scale * train_norm, V=V, inverse=True)
    rggb_img = recovery(ori_shape, out, size)
    img = rggb_img.transpose(1, 2, 0)
    new_raw = unpack(img)
    train_raw.raw_image_visible[:] = new_raw
    predict_rgb = train_raw.postprocess(use_camera_wb=True)
    predict_psnr = peak_signal_noise_ratio(predict_rgb, gt_rgb, data_range=256.)
    return ori_psnr, predict_psnr, predict_rgb


def gif_predict(train_path, gt_path, cuda=False, size=(2016, 3024), inp_scale=256, batch_size=1):
    ori_psnrs = []
    predict_psnrs = []
    imgs = []
    for i in range(len(train_path)):
        ori_psnr, predict_psnr, predict_rgb = new_predict(train_path[i], gt_path[i], cuda,
                                                          size, inp_scale, batch_size)
        ori_psnrs.append(ori_psnr)
        predict_psnrs.append(predict_psnr)
        imgs.append(predict_rgb)
    mimsave('animation_after.gif', imgs, "GIF", duration=0.5)
    return ori_psnrs, predict_psnrs


if __name__ == "__main__":
    ori_psnr, cur_psnr, predict_rgb = new_predict()
    print('original psnr is {:.2f} dB'.format(ori_psnr))
    print('current psnr is {:.2f} dB'.format(cur_psnr))
    # gt_raw = rawpy.imread("kaggle_img/gt_new.raw")
    # gt_rgb = gt_raw.postprocess(use_camera_wb=True)
    # cv2.imwrite("kaggle_img/kaggle_gt.jpg", gt_rgb)
    # train_raw = rawpy.imread('kaggle_img/input_new.raw')
    # train_rgb = train_raw.postprocess(use_camera_wb=True)
    # cv2.imwrite("kaggle_img/kaggle_train.jpg", train_rgb)
    cv2.imwrite("kaggle_img/kaggle_predict.jpg", predict_rgb)
