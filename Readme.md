# RAW image denoising demo

Reference: [Practical Deep Raw Image Denoising on Mobile Devices](https://arxiv.org/abs/2010.06935), 2020 CVPR

Necessary prerequisites: PyTorch / Megengine, rawpy, OpenCV (imageio is recommended.)

This is a reimplementation of a CVPR paper. The ideal environment is to load a full RAW image each time rather than crop images to a bunch of pieces
for training and prediction. This leads to a slightly high demand of computer's hardware. The backbone of the network 
is a U-Net based architecture. It's a light model with merely a million parameters. Model was trained with pretrained 
model by Megvii. Loss function was altered to L0 loss with gamma equals to 0.5 by default.

If the training samples are not enough, I believe over-fitting is a potential solution.

Sample result image is obtained without Cuda. For Cuda version, just set variable cuda to True.

