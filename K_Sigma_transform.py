import numpy as np
from typing import Tuple
import torch as meg
from scipy.optimize import leastsq
from load_data import getRawInfo
from utils import rgb2gray


class KSigma:
    def __init__(self, K_coeff: Tuple[float, float],
                 B_coeff: Tuple[float, float, float],
                 anchor: float,
                 V: float = 65024.0):  # 16-bit sensor with black level is 512 per channel
        self.K = np.poly1d(K_coeff)
        self.Sigma = np.poly1d(B_coeff)
        self.anchor = anchor
        self.V = V

    def __call__(self, img_01, iso: float, inverse=False):
        k, sigma = self.K(iso), self.Sigma(iso)
        k_a, sigma_a = self.K(self.anchor), self.Sigma(self.anchor)

        cvt_k = k_a / k
        cvt_b = (sigma / (k ** 2) - sigma_a / (k_a ** 2)) * k_a

        img = img_01 * self.V

        if not inverse:
            img = img * cvt_k + cvt_b
        else:
            img = (img - cvt_b) / cvt_k

        return img / self.V


def cal_kb(rgbs):
    def fun(p, x):
        k, b = p
        return k * x + b

    def error(p, x, y):
        return fun(p, x) - y
    grayscales = rgb2gray(rgbs)
    mean = grayscales.mean(dim=[0,1])
    var = grayscales.var(dim=[0,1], unbiased=True)
    mean = mean.flatten().numpy()
    var = var.flatten().numpy()
    init_k = (var[0]-var[1]) / (mean[0]-mean[1])
    init_b = var[0] - init_k * mean[0]
    p0 = np.array([init_k, init_b])
    param = leastsq(error, p0, args=(mean, var))
    k, b = param[0]
    return k, b


def ksigmaTransform(rggb, V=65024, inverse=False):
    K_coeff = (0.0005995267, 0.00868861)
    B_coeff = (7.11772e-7, 6.514934e-4, 0.11492713)
    anchor = 1600
    ksigma = KSigma(K_coeff, B_coeff, anchor, V)
    return ksigma(rggb, getRawInfo()['ISO'], inverse=inverse)


if __name__ == "__main__":
    ksigma = KSigma(
        (0.0005995267, 0.00868861),
        (7.11772e-7, 6.514934e-4, 0.11492713),
        1600
    )
    inputs = [meg.ones((4, 10, 10)) for _ in range(10)]
    inputs = meg.stack(inputs)
    for inp in inputs:
        print(inp.shape)