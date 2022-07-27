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
_** You have to change PyTorch version if you are using different version of CUDA_  
Find your wanted version of CUDA here and install following the Docs: 
https://developer.nvidia.com/cuda-toolkit-archive  

### After setting up CUDA  
#### Install Pytorch
Fidn your wanted version here: https://pytorch.org/get-started/locally/
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
```
pip install -r requirements.txt
```

## Training your model
### Data preprocess - preprocess_training.py
#### Usage
```
py preprocess_training.py --train_A_dir <data_directory_person_A> --train_B_dir <data_directory_person_B> --cache_folder <cache_directory>
```
#### Example
```
py preprocess_training.py --train_A_dir .\data\20-7-clean\ --train_B_dir .\data\20-7-tts\ --cache_folder ./cache/
```

## Start the training - train.py
#### Usage
```
python train.py --logf0s_normalization <path_to_logf0s_cache> --mcep_normalization <path_to_mcep_cache> --coded_sps_A_norm <path_to_coded_sps_A_cache> --coded_sps_B_norm <path_to_coded_sps_A_cache> --model_checkpoint <path_to_model_checkpoint_directory> --validation_A_dir <path_to_data_for_validation_A> --output_A_dir <path_to_output_A> --validation_B_dir <path_to_data_for_validation_B> --output_B_dir <path_to_output_B>
```
#### Example
```
python train.py --logf0s_normalization ./cache/logf0s_normalization.npz --mcep_normalization ./cache/mcep_normalization.npz --coded_sps_A_norm ./cache/coded_sps_A_norm.pickle --coded_sps_B_norm ./cache/coded_sps_B_norm.pickle --model_checkpoint ./model_checkpoint/ --validation_A_dir ./data/10_min_A/ --output_A_dir ./converted_sound/10_min_valid_A --validation_B_dir ./data/10_min_B/ --output_B_dir ./converted_sound/10_min_valid_B
```
#### Resume an old training
Adds:
```
--resume_training_at <path_to_checkpoint>
```
#### Do no training but voice clone only (need to provide model checkpoint)
Adds
```
--vc_only True --resume_training_at <path_to_checkpoint>
```
