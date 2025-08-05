


### ESPANet: Efficient Spatial Parameter-Free Attention Approximation Network for Single Image Super Resolution

## Install
Create a conda enviroment:
````
ENV_NAME="esapnet"
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
````
Run following script to install the dependencies:
````
bash install.sh
````


## Checkpoints
Pre-trained checkpoints are available in [here](https://drive.google.com/drive/folders/1oyWNSlQTpPwbZpSADxpskdZWqt_Pjm73?usp=sharing). Place the checkpoints in `checkpoints/`.


##### **Testing**
For testing the pre-trained checkpoints please use following commands. Replace `[TEST OPT YML]` with the path to the corresponding option file.
`````
CUDA_VISIBLE_DEVICES=0 python basicsr/test.py -opt [TEST OPT YML]
`````

##### **Training**
For single-GPU training use the following commands. Replace `[TRAIN OPT YML]` with the path to the corresponding option file.
`````
torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt [TRAIN OPT YML] --launcher pytorch

`````
OR

`````
CUDA_VISIBLE_DEVICES=0 basicsr/train.py -opt [TRAIN OPT YML]

# Example
CUDA_VISIBLE_DEVICES=0 basicsr/train.py -opt options/train/espanet_x2.yml

`````

## Miscellaneous/Notes
The current code architecture is named ESCANet. The structure is equivalent to ESPANet. The naming is changed for better understanding. 

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
