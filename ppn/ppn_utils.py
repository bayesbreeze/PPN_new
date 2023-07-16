# this file is adapted from https://github.com/yang-song/score_inverse_problems/blob/main/cs.py
# import numpy as np
import torch as th

def get_cartesian_mask(shape, n_keep=30):
    # shape [Tuple]: (H, W)
    size = shape[0]
    center_fraction = n_keep / 1000
    acceleration = size / n_keep

    num_rows, num_cols = shape[0], shape[1]
    num_low_freqs = int(round(num_cols * center_fraction))

    # create the mask
    mask = th.zeros((num_rows, num_cols), dtype=th.float32)
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[:, pad: pad + num_low_freqs] = True

    # determine acceleration rate by adjusting for the number of low frequencies
    adjusted_accel = (acceleration * (num_low_freqs - num_cols)) / (
        num_low_freqs * acceleration - num_cols
    )

    offset = round(adjusted_accel) // 2

    accel_samples = th.arange(offset, num_cols - 1, adjusted_accel)
    accel_samples = th.round(accel_samples).to(th.long)
    mask[:, accel_samples] = True

    # print("====>>>>> acc: %.5f, adjust: %.5f" % (acceleration, adjusted_accel))

    return mask


def get_kspace(img, axes):
  shape = img.shape[axes[0]]
  return th.fft.fftshift(
      th.fft.fftn(th.fft.ifftshift(
          img, dim=axes
      ), dim=axes),
      dim=axes
  ) / shape


def kspace_to_image(kspace, axes):
  shape = kspace.shape[axes[0]]
  return th.fft.fftshift(
      th.fft.ifftn(th.fft.ifftshift(
          kspace, dim=axes
      ), dim=axes),
      dim=axes
  ) * shape

to_space = lambda x: get_kspace(x, (2, 3)) # x: b c w h
from_space = lambda x: kspace_to_image(x, (2, 3)).real

def get_noisy_known(known, alpha, beta):
    z = th.rand_like(known)
    return alpha * known + beta * to_space(z)

def merge_known_with_mask(x_space, known, mask, coeff=1.):
    return known * mask * coeff + x_space * (1. - mask * coeff)



### test



