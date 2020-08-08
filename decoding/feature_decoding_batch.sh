#!/bin/sh

set -Ceu

networks="caffe/AlexNet
          caffe/DenseNet_121
          caffe/DenseNet_161
          caffe/DenseNet_169
          caffe/DenseNet_201
          caffe/InceptionResNet-v2
          caffe/SqueeseNet
          caffe/SqueeseNet1.0
          caffe/VGG-F
          caffe/VGG-M
          caffe/VGG-S
          caffe/VGG16_nonaka
          caffe/VGG19_nonaka
          pytorch/CORnet_R
          pytorch/CORnet_S
          pytorch/CORnet_Z
          pytorch/resnet18
          pytorch/resnet34
          tensorflow/inception_v1
          tensorflow/inception_v2
          tensorflow/inception_v3
          tensorflow/inception_v4
          tensorflow/mobilenet_v2_1.4_224
          tensorflow/nasnet_large
          tensorflow/nasnet_mobile
          tensorflow/pnasnet_large
          tensorflow/resnet_v2_101
          tensorflow/resnet_v2_152
          tensorflow/resnet_v2_50"

for net in $networks; do
    echo "Network: $net"
    python feature_decoding_train.py --net $net
    python feature_decoding_predict.py --net $net
done

