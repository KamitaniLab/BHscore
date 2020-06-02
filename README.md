# Brain Hierarchy Score

This repository contains data and scripts to repproduce the results from the paper ... by Soma Nonaka, Kei Majima, Shuntaro C. Aoki, and Yukiyasu Kamitani. 

## Requirements
* BdPy (pip install bdpy)
* FastL2LiR (pip install fastl2lir)
* NumPy
* SciPy
* Pillow
* hdf5storage
* PyTorch, Caffe or TensorFlow


## Calculate BH score with your DNNs

### 1. Preparation

#### fMRI data

We used fMRI data from [a previous study from our lab](https://github.com/KamitaniLab/DeepImageReconstruction). To download fMRI data, run `data/download.sh`. About xxGB storage is needed. You can also download fMRI data manually from [figshare](https://figshare.com/articles/Deep_Image_Reconstruction/7033577) (In that case, place the downloaded files to `data/fmri_data`). 

#### Visual images

Images from the ILSVRC2012 dataset were used in our study.
For copyright reasons, we do not distribute the images.
You can request us the images via <https://forms.gle/ujvA34948Xg49jdn9>.

### 2. DNN feature extraction

Modify settings in `feature_extraction/extract_features_(caffe | pytorch).py` based on your DNN. 
The, run the script. Features from all layers are automatically extracted. Extracted features are stored in `./data/features/<network-name>`. 
```
cd feature_extraction
python extract_features_caffe.py (or feature_extraction/extract_features_pytorch.py)
```
*Optional*: If you want to extract features in a part of layers, please modify settings in the script.   
We recommend <https://github.com/tomrunia/TF_FeatureExtraction> to extract features from Tensorflow models.


### 3. Decoding analysis

*Optional*: open `settings.json` and configure settings below if you need to change these settings. 
* ROI (default: V1, V2, V3, V4, HVC)
* subjects (default: sub-01, sub-02, sub-03)
* DNN layers (If not specified, all layers are used)
* Number of DNN units to be predicted (default: 1000)
* other hyper-parameters (regularization term for linear regression, number of voxels to be selected, etc.)

Run the following scripts. 
```
python feature_prediction_train.py
python feature_prediction_test.py
```
The predicted features and prediction accuracies are stored in `./decoding/results/`. 
To speed up the computation, run the script parallelly. 

### 4. Calculate BH score

To compute BH score, run 
```
python calc_bhscore.py
``` 
The result will be stored as a file named `bhscores.mat`. 
*Optional*: Please edit `settings.json` if you want to use different settings of ROI, DNN layers, etc. 


## Reproduction of our results
We provide scripts that reproduce main figures in the original paper (see `analysis/`). 

## Misc
* If your caffe prototxt file uses in-place computation, use feature_extraction/modify_layer_name.py to access activations of intermidiate layers. 
