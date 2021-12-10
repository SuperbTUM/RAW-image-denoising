# RAW image denoising demo

### Introduction

Reference: [Practical Deep Raw Image Denoising on Mobile Devices](https://arxiv.org/abs/2010.06935), 2020 ECCV

[Learning Raw Image Denoising with Bayer Pattern Unification and Bayer Preserving Augmentation](https://arxiv.org/abs/1904.12945), 2019 CVPR	

Data transformation and backbone are what 2020 paper had done. Data augmentation is what 2019 paper had done. The backbone of the network is a U-Net based architecture. It's a light model with merely a million parameters. Model was trained with pretrained model by Megvii.

### Codes review

Necessary prerequisites: PyTorch / Megengine, rawpy, scikit-image, OpenCV (imageio is recommended.)

Everything starts at `main.py`. You may notice there are two checkpoints, one is pre-trained model from Megvii, the other comes from my own training process. If it is the first time you train the network, you can load the pre-trained model called `torch_pretrained.ckp`. Once you have a valid saved network, you can load your own network. It's time-saving and not very reliable on the equipment. You can train the network with cropped images in dedicated size or complete images. Images mentioned here are all in rggb format. 

We need to pay attention to K-Sigma transformation. Frankly speaking, K-Sigma transformation takes Poisson inputs and Gaussian noises and approximates them as a new Gaussian distribution with linear transformation as outputs. This transformation enhances network's robustness on camera meta data. There are four key parameters in this transformation. I don't know how to get k and sigma though I understand these comes from linear regression (I don't know how the static grayscale chart functions in terms of luminance estimation and I guess it's just a reference.) but anchor is an empirical parameter with 1600 by default and V equals to theoretical maximum value of RAW image (bit width of sensor) minus black level.

With epoch=30, network trained with complete image can reach 39.6 dB. Note that the original images pair comparison is 28.2 dB in PSNR. If the training samples are not enough, just like this demo, I believe over-fitting is a potential solution.

From another dataset I downloaded a pair of images and with this network, PSNR was raised from 30.9 dB to 37.3 dB. You can also do a series of image denoising to obtain gif.

I made another attempt by modifying the model. I tried to integrate SE block (just like self-attention) into residual blocks in both encoder and decoder. I also had issues towards the up sampling block. Why using deconvolution every time? I tried to replace it with pure up sample and convolution layer. The outcome may not be satisfactory as you do not have pre-trained models. It may take a lot of time to obtain a well-trained model, or maybe I am wrong.

Sample result image is obtained without Cuda. For Cuda version, just set variable cuda to True.

An updated version may be released if there is any progress.

