#!/bin/bash

set -Ceu

[ -e ./fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5 ] || wget -O ./fmri/sub-01_perceptionNaturalImageTraining_original_VC.h5 https://ndownloader.figshare.com/files/14830643
[ -e ./fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5 ] || wget -O ./fmri/sub-02_perceptionNaturalImageTraining_original_VC.h5 https://ndownloader.figshare.com/files/14830712
[ -e ./fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5 ] || wget -O ./fmri/sub-03_perceptionNaturalImageTraining_original_VC.h5 https://ndownloader.figshare.com/files/14830862
[ -e ./fmri/sub-01_perceptionNaturalImageTest_original_VC.h5 ] || wget -O ./fmri/sub-01_perceptionNaturalImageTest_original_VC.h5 https://ndownloader.figshare.com/files/14830631
[ -e ./fmri/sub-02_perceptionNaturalImageTest_original_VC.h5 ] || wget -O ./fmri/sub-02_perceptionNaturalImageTest_original_VC.h5 https://ndownloader.figshare.com/files/14830697
[ -e ./fmri/sub-03_perceptionNaturalImageTest_original_VC.h5 ] || wget -O ./fmri/sub-03_perceptionNaturalImageTest_original_VC.h5 https://ndownloader.figshare.com/files/14830856
