# RAW image denoising demo

### Introduction

Reference: [Practical Deep Raw Image Denoising on Mobile Devices](https://arxiv.org/abs/2010.06935), 2020 CVPR

This is a reimplementation of a CVPR paper. The backbone of the network is a U-Net based architecture. It's a light model with merely a million parameters. Model was trained with pretrained model by Megvii.

### Codes review

Necessary prerequisites: PyTorch / Megengine, rawpy, OpenCV (imageio is recommended.)

Everything starts at `main.py`. You may notice there are two checkpoints, one is pretrained model from Megvii, the other comes from my own training process. You can train the network with cropped images in dedicated size or complete images. Images mentioned here are all in rggb format. If the training samples are not enough, just like this demo, I believe over-fitting is a potential solution.

I made another attempt by modifying the model. I tried to integrate SE block (just like self-attention) into residual blocks in both encoder and decoder. The outcome may not be satisfactory.

Sample result image is obtained without Cuda. For Cuda version, just set variable cuda to True.

An updated version is coming soon.
