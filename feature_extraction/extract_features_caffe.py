'''Extract DNN features from images using Caffe.'''


from __future__ import absolute_import, division, print_function

import glob
import os

import PIL
import caffe
import numpy as np
from bdpy.dataform import save_array
from scipy.misc import imresize
from tqdm import tqdm


## Settings #############################################################################

network_name = 'example_network'
image_dir = './imagenet_test_images'
prototxt = 'example.prototxt'
caffemodel = 'example.caffemodel'
mean_image = np.float32([104., 117., 123.])
use_gpu = False

output_dir = os.path.join('../data/features', network_name)

############################################################################################


def extract_features(image_dir, prototxt, caffemodel, mean_image, output_dir, layers=None, use_gpu=False):

    # GPU usage settings
    if use_gpu == True:
        caffe.set_mode_gpu()
        caffe.set_device(0)
    else:
        caffe.set_mode_cpu()

    # Directory setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get mean image
    if os.path.isfile(mean_image):
        mean_image_full = np.load(mean_image)
        mean_image = np.float32([mean_image_full[0].mean(
        ), mean_image_full[1].mean(), mean_image_full[2].mean()])

    # Load network
    net = caffe.Classifier(prototxt, caffemodel,
                           mean=mean_image, channel_swap=(2, 1, 0))

    image_size = net.blobs['data'].data.shape[-2:]
    print('Input image size: %d x %d' % image_size)

    # list up all layers if not specified
    if layers is None:
        layers = net.blobs.keys()
        layers.remove('data')
        layers = [layer for layer in layers if 'split' not in layer]

    print('Layers: ' + ', '.join(layers))

    # Extract features
    image_files = glob.glob(os.path.join(image_dir))
    for imgf in tqdm(image_files):
        # print('Image:  %s' % imgf)

        # Open an image
        try:
            img = PIL.Image.open(imgf)
        except:
            print('Error: cannot load file: ', imgf)
            exit()

        if img.mode != 'RGB':
            img.convert('RGB')

        img = imresize(img, image_size, interp='bicubic')

        mean_img = net.transformer.mean['data']

        # Convert monotone to RGB
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)

        img = np.float32(np.transpose(img, (2, 0, 1))[
                         ::-1]) - np.reshape(mean_img, (3, 1, 1))

        # Forwarding
        net.blobs['data'].reshape(1, 3, img.shape[1], img.shape[2])
        net.blobs['data'].data[0] = img
        net.forward()

        # Save features
        for lay in layers:
            # print('Layer: %s' % lay)
            # print('  feature shape = (%s)' % ', '.join([str(s) for s in feat.shape]))
            save_dir = os.path.join(output_dir, lay.replace('/', ':'))
            save_file = os.path.join(save_dir, os.path.splitext(
                os.path.basename(imgf))[0] + '.mat')

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            feat = net.blobs[lay].data.copy()

            save_array(save_file, feat, key='feat',
                       dtype=np.float32, sparse=False)
            # print('Saved %s' % save_file)

    print('Done!')


if __name__ == '__main__':
    extract_features(image_dir, prototxt, caffemodel,
                     mean_image, output_dir, use_gpu)
