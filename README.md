# scBeacon

This is the official implementation of scBeacon

## Table of Contents

- [Requirements](#requirements)
- [How to train](#how-to-train)
- [Acknowledgement](#acknowledgement)

## Requirements
* Python >= 3.6
* Python side-packages:   
-- pytorch >= 1.9.0  
-- numpy >= 1.19.2     
-- pandas>=1.1.5   
-- scanpy >= 1.7.2  
-- leidenalg>=0.8.4  
-- tqdm >= 4.61.1  
-- scikit-learn>=0.24.2  
-- umap-learn>=0.5.1  
-- matplotlib >= 3.3.4   
-- seaborn >= 0.11.0   

## Installation 

We recommend to use virture environment to install the packages. After successfully installing Anaconda/Miniconda, create an environment by the following: 

```shell
conda create -n scBeacon python=3.6
```

Then activate the environment by: 

```shell
conda activate scBeacon
```

Then use either **pip install** or **conda install** to install the required packages in the environment.

### Pytorch install
scBeacon is built based on Pytorch and supporting both CPU or GPU. Make sure you have Pytorch (>= 1.9.0) installed in your virtual environment. If not, please visit [Pytorch](https://pytorch.org/) and install the appropriate version. 

## How to train

### Hyperparameters
You can modify the Hyperparameters in config_label.py. 

### Train
You can train with:
```shell
python train_labelv4.py
```

## Acknowledgement
This project is built upon the foundations of https://github.com/doraadong/UNIFAN

# License 
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
