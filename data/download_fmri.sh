#!/bin/bash

set -Ceu

[ -e ./fmri/sub-01_perceptionNaturalImageTest_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-01_perceptionNaturalImageTest_VC_v2.h5 https://ndownloader.figshare.com/files/33900890
[ -e ./fmri/sub-01_perceptionNaturalImageTraining_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-01_perceptionNaturalImageTraining_VC_v2.h5 https://ndownloader.figshare.com/files/33900893
[ -e ./fmri/sub-02_perceptionNaturalImageTest_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-02_perceptionNaturalImageTest_VC_v2.h5 https://ndownloader.figshare.com/files/33900896
[ -e ./fmri/sub-02_perceptionNaturalImageTraining_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-02_perceptionNaturalImageTraining_VC_v2.h5 https://ndownloader.figshare.com/files/33900899
[ -e ./fmri/sub-03_perceptionNaturalImageTest_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-03_perceptionNaturalImageTest_VC_v2.h5 https://ndownloader.figshare.com/files/33900902
[ -e ./fmri/sub-03_perceptionNaturalImageTraining_VC_v2.h5 ] || wget --no-check-certificate -O ./fmri/sub-03_perceptionNaturalImageTraining_VC_v2.h5 https://ndownloader.figshare.com/files/33900908
