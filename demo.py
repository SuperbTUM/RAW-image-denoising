from utils import *
import cv2
from skimage.metrics import peak_signal_noise_ratio


if __name__ == '__main__':
    raw = rawpy.imread('img_data/groundtruth.ARW')
    gt_rgb = raw.postprocess(use_camera_wb=True)
    raw_train = rawpy.imread('img_data/test.ARW')
    train_rgb = raw_train.postprocess(use_camera_wb=True)
    psnr_before = peak_signal_noise_ratio(train_rgb, gt_rgb, data_range=256.)
    print('psnr before processing is {} dB'.format(psnr_before))

    my_rgb = cv2.imread('result_image.jpg')
    my_rgb = cv2.cvtColor(my_rgb, cv2.COLOR_BGR2RGB)

    psnr_after = peak_signal_noise_ratio(my_rgb, gt_rgb, data_range=256.)
    print("psnr after processing is {} dB".format(psnr_after))
