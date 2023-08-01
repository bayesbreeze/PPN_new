import numpy as np
import matplotlib.pyplot as plt
import torch 
import h5py
import sigpy.mri as mr

fn="/Users/John/Downloads/multicoil_train/file1000584.h5"
h5file = h5py.File(fn, 'r') 
ks=h5file['kspace']  #(41, 15, 640, 368)


def img_normalize_complex(complex_image):
    imgs = np.abs(complex_image)
    v_min = imgs.min()
    imgs -= v_min
    v_max = np.percentile(imgs.reshape(-1), 99.8)
    imgs[imgs>v_max]=v_max
    imgs /= v_max

    return imgs * np.exp(1j * np.angle(complex_image))

def img_normalize(imgs):
    v_min = imgs.min()
    imgs -= v_min # make sure the value start from 0
    v_max = np.percentile(imgs.reshape(-1), 99.8)
    imgs[imgs>v_max]=v_max # clip max value to 99.8%
    imgs /= v_max # imgs is in [0,1]
    return imgs

def cut_center(imgs, w=320):
    height, width = imgs.shape[-2], imgs.shape[-1]
    start_y = (height - w) // 2
    start_x = (width - w) // 2
    return imgs[..., start_y:start_y+w, start_x:start_x+w]

def get_kspace_np(imgs, axes):
    return np.fft.fftshift(
        np.fft.fftn(
            np.fft.ifftshift(imgs, axes=axes), 
            axes=axes, norm="ortho"
        ), axes=axes,
    )

def kspace_to_image_np(kspace, axes):
  return np.fft.fftshift(
        np.fft.ifftn(
            np.fft.ifftshift(kspace, axes=axes), 
            axes=axes, norm="ortho"
        ), axes=axes,
    )

to_space = lambda imgs: get_kspace_np(imgs, axes=(-2,-1)) 
from_space = lambda ks: kspace_to_image_np(ks, axes=(-2,-1)) 


def rss_complex(d, axis=-3): 
    dd = np.stack([d.real, d.imag], axis=-1)
    rt = np.sqrt((dd**2).sum(axis=-1).sum(axis=axis))
    return np.expand_dims(rt, axis=axis)


def print_All(imgs):
    fig, axs = plt.subplots(3,3,figsize=(10,10))
    for (img, ax) in zip(imgs, axs.flat):
        ax.imshow(np.abs(img).squeeze(),cmap="gray")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def calc_sens(imgs):
    kss = to_space(imgs)
    sens=[mr.app.EspiritCalib(ks).run() for ks in kss]
    return np.stack(sens)


imgs_kspace=from_space(ks)
imgs_cut=cut_center(imgs_kspace)
imgs_normalized=img_normalize_complex(imgs_cut)

all_imgs=imgs_normalized[15::3].astype(np.complex64)
all_sens=calc_sens(all_imgs) # !!!time consuming
all_sens=all_sens.astype(np.complex64)

np.savez("testsamples", all_imgs=all_imgs, all_sens=all_sens)

# # print rss files
# imgs_rss = rss_complex(imgs_normalized) 
# print_All(imgs_rss[15::3])
# print_All(all_imgs[0])



