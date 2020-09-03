#!/bin/bash

set -Ceu

[ -e ./features/ImageNetTraining/caffe/ ]      || (wget --no-check-certificate -O ./features-ImageNetTraining-caffe.zip      https://ndownloader.figshare.com/files/24172706 && unzip ./features-ImageNetTraining-caffe.zip)
[ -e ./features/ImageNetTraining/pytorch/ ]    || (wget --no-check-certificate -O ./features-ImageNetTraining-pytorch.zip    https://ndownloader.figshare.com/files/24172862 && unzip ./features-ImageNetTraining-pytorch.zip)
[ -e ./features/ImageNetTraining/tensorflow/ ] || (wget --no-check-certificate -O ./features-ImageNetTraining-tensorflow.zip https://ndownloader.figshare.com/files/24172895 && unzip ./features-ImageNetTraining-tensorflow.zip)

[ -e ./features/ImageNetTest/caffe/ ]      || (wget --no-check-certificate -O ./features-ImageNetTest-caffe.zip      https://ndownloader.figshare.com/files/24172679 && unzip ./features-ImageNetTest-caffe.zip)
[ -e ./features/ImageNetTest/pytorch/ ]    || (wget --no-check-certificate -O ./features-ImageNetTest-pytorch.zip    https://ndownloader.figshare.com/files/24172691 && unzip ./features-ImageNetTest-pytorch.zip)
[ -e ./features/ImageNetTest/tensorflow/ ] || (wget --no-check-certificate -O ./features-ImageNetTest-tensorflow.zip https://ndownloader.figshare.com/files/24172697 && unzip ./features-ImageNetTest-tensorflow.zip)
