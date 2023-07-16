"""fastmri_knee dataset."""

import tensorflow_datasets as tfds
import h5py
import numpy as np
import glob
import tarfile
import my_h5_getter as myh5


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for fastmri_knee dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # TODO(fastmri_knee): Specifies the tfds.core.DatasetInfo object
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(320, 320, 1))
            }),
            supervised_keys=None,  # Set to `None` to disable
            homepage='https://dataset-homepage/',
        )

    def img_normalize(self, imgs):
        v_min = imgs.min()
        imgs -= v_min # make sure the value start from 0
        v_max = np.percentile(imgs.reshape(-1), 99.8)
        imgs[imgs>v_max]=v_max # clip max value to 99.8%
        imgs /= v_max # imgs is in [0,1]
        return imgs

    def list_h5_from_tar(self, path):
        with tarfile.open(path, 'r') as tar:
            # Process each h5 file within the tar file
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.h5'):
                    h5_file = tar.extractfile(member)
                    with h5py.File(h5_file, 'r') as h5:
                        yield h5
                    h5_file.close()
    
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        path = "/Users/John/Downloads"
        return {
            'train': self._generate_examples('/Users/John/Downloads/singlecoil_train'),
            'val': self._generate_examples('/Users/John/Downloads/singlecoil_val'),
        }

    # # knee_singlecoil_train.tar 
    # def _generate_examples(self, path):
    #     count = -1
    #     for h5 in self.list_h5_from_tar(path):
    #         imgs = h5['reconstruction_rss'][()][5:-3] 
    #         imgs = self.img_normalize(imgs)

    #         for img in imgs:
    #             count += 1
    #             yield count, {
    #                 'image': np.clip(img[..., None] * 255., 0.0, 255.).astype(np.uint8)
    #             }
    
    # pixz -d < knee_singlecoil_train.tar  | tar xv
    # pixz -d < knee_singlecoil_val.tar  | tar xv
    def _generate_examples(self, path):
        count = -1
        for fn in myh5.get_h5_imgs(path, file_end=".h5", needDel=True):
            try:
                with h5py.File(fn, 'r') as h5:
                    imgs = h5['reconstruction_rss'][()]
                    imgs = self.img_normalize(imgs[5:-3])
                    for img in imgs:
                        count += 1
                        yield count, {
                            'image': np.clip(img[..., None] * 255., 0.0, 255.).astype(np.uint8)
                        }
            except Exception as e:
                print("@@@ failed to process %s, with error: %s" %(fn, str(e)))

# ds, info = tfds.load('fastmri_knee', split='train', shuffle_files=True, 
#                     as_supervised=False,  with_info=True)
# tfds.show_examples(ds, info)  # show examples

# import matplotlib.pyplot as plt

# ds, info = tfds.load('fastmri_knee', split='train', shuffle_files=True, 
#                     as_supervised=False, with_info=True)
# fig, axes = plt.subplots(nrows=3, ncols=3)
# for img,axe in zip(tfds.as_numpy(ds.take(9)), axes.flatten()):
#     axe.imshow(img['image'], cmap='gray')
#     axe.axis('off')
# plt.tight_layout()
# # plt.show()
# plt.savefig("ttt.jpg")
# plt.close()