# RAW image denoising demo

### Introduction

Reference: [Practical Deep Raw Image Denoising on Mobile Devices](https://arxiv.org/abs/2010.06935), 2020 CVPR

This is a reimplementation of a CVPR paper. The backbone of the network is a U-Net based architecture. It's a light model with merely a million parameters. Model was trained with pretrained model by Megvii.

### Codes review

Necessary prerequisites: PyTorch / Megengine, rawpy, OpenCV (imageio is recommended.)

Everything starts at `main.py`. You may notice there are two checkpoints, one is pre-trained model from Megvii, the other comes from my own training process. If it is the first time you train the network, you can load the pre-trained model called `torch_pretrained.ckp`. Once you have a valid saved network, you can load your own network. It's time-saving and not very reliable on the equipment. You can train the network with cropped images in dedicated size or complete images. Images mentioned here are all in rggb format. 

There is an interesting phenomena saying network trained with cropped images performs better than that with complete image. It's not difficult to understand but be aware that this may not be applicable to other data. With epoch=10, network trained with cropped images can reach 38 dB in prediction stage, while network trained with complete image can only reach 30 dB. Note that the original images pair comparison is 28 dB in PSNR. However, with number of epoch is increasing, PSNR consistently increases and it can reach 35 dB (with complete image) after 60 epochs' training (I have not tested the limit). If the training samples are not enough, just like this demo, I believe over-fitting is a potential solution.

I made another attempt by modifying the model. I tried to integrate SE block (just like self-attention) into residual blocks in both encoder and decoder. I also had issues towards the up sampling block. Why using deconvolution every time? I tried to replace it with pure up sample and convolution layer. The outcome may not be satisfactory as you do not have pre-trained models. It may take a lot of time to obtain a well-trained model, or maybe I am wrong.

Sample result image is obtained without Cuda. For Cuda version, just set variable cuda to True.

An updated version is coming within a few days. (This is not a guarantee.)
