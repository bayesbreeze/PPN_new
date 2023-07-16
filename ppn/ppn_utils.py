# this file is adapted from https://github.com/yang-song/score_inverse_problems/blob/main/cs.py
import numpy as np
import torch as th
import piq
from guided_diffusion import dist_util, logger


# using pytorch tensor
def report_metrics_and_save(args, testset, samples):
    samples = th.clamp(samples, 0., 1.)
    psnr = np.array([piq.psnr(smp[None, ...], tgt[None, ...], data_range=1.).item() 
                for tgt, smp in zip(testset, samples)])
    ssim = np.array([piq.ssim(smp[None, ...], tgt[None, ...], data_range=1.).item() 
                for tgt, smp in zip(testset, samples)])
    report = "samples#%d_x%d_step%d_psnr_%.4f_%.4f_ssim_%.4f_%.4f"%(
        args.num_samples, args.acceleration, args.num_timesteps,
        psnr.mean(), psnr.std(), ssim.mean(), ssim.std())
    logger.log("report: ", report)

    save_path = "%s/%s.npz"%(logger.get_dir(), report)
    logger.log("saving to: ", save_path)
    np.savez(save_path, samples)

# load testset
def get_testset_and_mask(args,device="cpu"):
    imgs = np.load(args.testset_path)['all_imgs']
    imgs = imgs[:min(len(imgs), args.num_samples)] # limit sample number
    imgs = th.from_numpy(imgs[:,None,...]).float() # (1000, 1, 240, 240) b c w h
    imgs = imgs / 255.0   # convert to [0,1]
    mask = get_cartesian_mask(args.image_size, int(args.image_size/args.acceleration),device=device)
    return imgs, mask

def iter_testset(imgs, args):
    num_batches = int(np.ceil(len(imgs) / args.batch_size))
    for batch in range(num_batches):
        current_batch = imgs[batch * args.batch_size:
                            min((batch + 1) * args.batch_size, len(imgs))]
        yield current_batch

def get_cartesian_mask(size, n_keep=30, device="cpu"):
    # shape [Tuple]: (H, W)
    center_fraction = n_keep / 1000
    acceleration = size / n_keep

    num_rows, num_cols = size, size
    num_low_freqs = int(round(num_cols * center_fraction))

    # create the mask
    mask = th.zeros((num_rows, num_cols), dtype=th.float32, device=device)
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



