import numpy as np
from typing import Tuple
import torch as meg


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