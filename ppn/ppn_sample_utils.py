# this file is adapted from https://github.com/yang-song/score_inverse_problems/blob/main/cs.py
import numpy as np
import torch as th
import piq
from guided_diffusion import dist_util, logger
from einops import rearrange


# using pytorch tensor
def report_metrics_and_save(args, testset, samples):
    # TODO normalization is not correct for multicoil
    testset = th.clamp(testset, 0., 1.)
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
    np.savez(save_path, all_imgs=samples)

# load testset
def get_testset_and_mask(args):
    ds = np.load(args.testset_path)
    sens=None
    if 'all_sens' in ds:
        sens = ds['all_sens']
        sens = sens[:min(len(sens), args.num_samples)].astype(np.complex64)

    imgs = ds['all_imgs']
    imgs = imgs[:min(len(imgs), args.num_samples)].astype(np.complex64) # limit sample number
    # imgs = imgs[:,None,...] # (1000, 1, 240, 240) b c w h
    # imgs = imgs / 255.0   # convert to [0,1]  # TODO brats is uint8, multi-coil is float
    # create sampling mask
    mask = get_cartesian_mask(args.image_size, int(args.image_size/args.acceleration))
    mask = mask[None, None]  # (1,1,w,h)
    

    return imgs, sens, mask

def iter_testset(args, all_imgs, all_sens):
    num_batches = int(np.ceil(len(all_imgs) / args.batch_size))
    isMultiCoil = args.sampleType=="multicoil"
    for batch in range(num_batches):
        start=batch * args.batch_size
        end=min((batch + 1) * args.batch_size, len(all_imgs))
        if isMultiCoil:
            imgs = th.from_numpy(all_imgs[start:end]).to(dtype=th.complex64)
            sens = th.from_numpy(all_sens[start:end]).to(dtype=th.complex64)
        else:
            imgs = th.from_numpy(all_imgs[start:end]).float()
            sens = None
        yield imgs, sens

def get_cartesian_mask(size, n_keep=30):
    # shape [Tuple]: (H, W)
    center_fraction = n_keep / 1000
    acceleration = size / n_keep

    num_rows, num_cols = size, size
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

to_space = lambda x: get_kspace(x, (-2, -1)) # x: b c w h
from_space = lambda x: kspace_to_image(x, (-2, -1))

to_mc = lambda x: th.view_as_complex(rearrange(x, '(b c ch) 1 h w -> b c h w ch', c=15, ch=2).contiguous())
from_mc = lambda x: rearrange(th.view_as_real(x), 'b c h w ch -> (b c ch) 1 h w') 

def get_noisy_known(known, alpha, beta):
    z = th.rand_like(known)
    return alpha * known + beta * to_space(z)

def merge_known_with_mask(x_space, known, mask, coeff=1.):
    return known * mask * coeff + x_space * (1. - mask * coeff)

def rss_complex(d, axis=-3): 
    dd = th.stack([d.real, d.imag], dim=-1)
    return th.sqrt((dd**2).sum(dim=-1).sum(dim=axis)).unsqueeze(axis)



### test



