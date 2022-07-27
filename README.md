# CycleGAN-VC2-Doraemon-Cantonese
Original Repo: https://github.com/jackaduma/CycleGAN-VC2

## Repo structure:
```
.
└── Repo directory/
    ├── model_tf.py
    ├── preprocess.py
    ├── preprocess_training.py
    ├── train.py
    ├── trainingDataset.py
    ├── data/
    │   ├── sub-directory-1/
    │   │   ├── 1.wav
    │   │   ├── 2.wav
    │   │   └── ...
    │   └── sub-directory-2/
    │       ├── 1.wav
    │       ├── 2.wav
    │       └── ...
    ├── cache
    ├── converted_sound/
    │   ├── sub-directory-1/
    │   │   ├── 1.wav
    │   │   ├── 2.wav
    │   │   └── ...
    │   └── sub-directory-2/
    │       ├── 1.wav
    │       ├── 2.wav
    │       └── ...
    └── model_checkpoint/
        └── _CycleGAN_CheckPoint
```
## Install requirement

### Install & Setup CUDA before installing dependencies (CUDA 11.3 used in this repo)  
_** You have to change PyTorch version in requirements.txt if you are using different version of CUDA_  
Find your wanted version of CUDA here and install following the Docs: 
https://developer.nvidia.com/cuda-toolkit-archive  

### After setting up CUDA  
```
pip install -r requirements.txt
```
