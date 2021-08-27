from utils import *
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


if __name__ == '__main__':
    raw = rawpy.imread('img_data/groundtruth.ARW')
    gt_rgb = raw.postprocess(use_camera_wb=True)
    raw_train = rawpy.imread('img_data/test.ARW')
    train_rgb = raw_train.postprocess(use_camera_wb=True)
    psnr_before = peak_signal_noise_ratio(train_rgb, gt_rgb, data_range=256.)
    ssim_before = structural_similarity(train_rgb, gt_rgb, win_size=11, multichannel=True)
    print('psnr before processing is {:.2f} dB, ssim before processing is {:.4f}'.format(psnr_before, ssim_before))

    my_rgb = cv2.imread('result_image.jpg')
    my_rgb = cv2.cvtColor(my_rgb, cv2.COLOR_BGR2RGB)

    psnr_after = peak_signal_noise_ratio(my_rgb, gt_rgb, data_range=256.)
    ssim_after = structural_similarity(my_rgb, gt_rgb, win_size=11, multichannel=True)
    print("psnr after processing is {:.2f} dB, ssim after processing is {:.4f}".format(psnr_after, ssim_after))
