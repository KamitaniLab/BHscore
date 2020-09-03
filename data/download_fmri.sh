#!/bin/bash

set -Ceu

[ -e ./fmri/sub-01_perceptionNaturalImageTraining_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-01_perceptionNaturalImageTraining_VC_v2.h5 https://ndownloader.figshare.com/files/24153026
[ -e ./fmri/sub-02_perceptionNaturalImageTraining_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-02_perceptionNaturalImageTraining_VC_v2.h5 https://ndownloader.figshare.com/files/24153080
[ -e ./fmri/sub-03_perceptionNaturalImageTraining_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-03_perceptionNaturalImageTraining_VC_v2.h5 https://ndownloader.figshare.com/files/24153098
[ -e ./fmri/sub-01_perceptionNaturalImageTest_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-01_perceptionNaturalImageTest_VC_v2.h5 https://ndownloader.figshare.com/files/24153020
[ -e ./fmri/sub-02_perceptionNaturalImageTest_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-02_perceptionNaturalImageTest_VC_v2.h5 https://ndownloader.figshare.com/files/24153077
[ -e ./fmri/sub-03_perceptionNaturalImageTest_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-03_perceptionNaturalImageTest_VC_v2.h5 https://ndownloader.figshare.com/files/24153095
