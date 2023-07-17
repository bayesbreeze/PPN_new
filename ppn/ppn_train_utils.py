import tensorflow_datasets as tfds
import tensorflow as tf
import torch
tf.config.experimental.set_visible_devices([], "GPU")

def prepare_image(d_dict):
    img = d_dict['image']
    img = tf.cast(img, tf.float32) / 255. # each pixel is [0,1]
    img = tf.transpose(img, (2,0,1)) # b w h c => b c w h
    img = 2 * img - 1.0 
    return img

def data_wrapper(ds):
    for d in ds:
        yield torch.as_tensor(d.numpy(), dtype=torch.float32), {}

def get_datasets(dataset_name, batch_size):
    train, info = tfds.load(dataset_name, split='train', shuffle_files=True, 
                        as_supervised=False, with_info=True) # 26958
    train = train.map(prepare_image)
    train = train.repeat().shuffle(512)
    train = train.batch(batch_size).prefetch(-1)
    print("[dataset] train number: %d" % (info.splits['train'].num_examples))

    val, info= tfds.load(dataset_name, split='val[:30%]', shuffle_files=True, 
                        as_supervised=False, with_info=True) # 1662 = 5543 * 0.3
    # val = val.map(prepare_image)
    val = val.repeat().shuffle(512)
    val = val.batch(batch_size, drop_remainder=True).prefetch(-1)
    print("[dataset] validate number: %d" % (info.splits['val'].num_examples * 0.3))
    
    return data_wrapper(train), data_wrapper(val)


