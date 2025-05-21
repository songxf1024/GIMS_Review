# Image Matching System Based on Adaptive Graph Construction and Graph Neural Network

---
    

## Directory Structure
Below is a brief overview of the directory structure and the role of each component:
```
gims/                     # Source code for GIMS framework
├── assets/               # Example images for demo and visualization
├── carhynet/             # CAR-HyNet descriptor implementation
├── configs/              # Training configuration files
├── models/               # AGC and GNN matcher
├── tools/                # Some auxiliary tools
├── utils/                # Common utility functions and helpers
├── weights/              # Pretrained weights for GIMS and CAR-HyNet
├── generate_pairs.py     # Script to generate image pairs for training/evaluation
├── train.py              # Training entry point
├── eval_homography.py    # Homography estimation and AUC evaluation
├── eval_match.py         # Matching evaluation and statistics
└── requirements.txt      # Python dependencies
```

> The weights of GIMS and CAR-HyNet can be downloaded at [Google Drive](https://kutt.it/gims)

## Setup
### Environment
#### Hardware Dependencies
> The following configurations are recommended and not mandatory.
- X86-CPU machine (>=16 GB RAM) 
- Nvidia GPUs (>10 GB each)

#### Software Dependencies
- Ubuntu 20.04 LTS
- Python 3.9.21
- CUDA 11.7
- PyTorch 2.0.1
- DGL 1.1.2

### Installation
Running the following commands will create the virtual environment and install the dependencies. `conda` can be downloaded from [anaconda](https://www.anaconda.com/download/success). 
```bash
conda create -n gims python=3.9.12
conda activate gims

pip install -r requirements
```

### Datasets
- We use [COCO2017](https://cocodataset.org/#download) for training. Download the 'train2017', 'val2017', and 'annotations' folder and put the folder path in the config file.
- We use [COCO2017 (Test images)](http://images.cocodataset.org/zips/test2017.zip), [DIML RGB-D](https://dimlrgbd.github.io/), and [Oxford-Affine](https://www.robots.ox.ac.uk/~vgg/research/affine/) for evaluation. 

## Usage
### Experiment Customization
Adjust configurations in `configs/coco_config.yaml` to customize  train params, optimizer params, and dataset params. In general, you only need to modify `dataset_params.dataset_path`.

### Train the Model
Running the following command will start train the model.
```bash
python train.py --gpus="0" --limit=-1 --name=gims
```
The output in the console will be like:
```
GPU 0: NVIDIA GeForce RTX 3090
==> CAR-HyNet successfully loaded pre-trained network.
Optimizer groups: 139 .bias, 120 conv.weight, 23 other
loading annotations into memory...
Done (t=18.04s)
creating index...
index created!
Started training for 2 epochs
Number of batches: 118286
Chang learning rate to 0.0001
Started epoch: 1 in rank -1
Epoch   gpu_mem   Iteration   PosLoss   NegLoss   TotLoss     Dtime     Ptime     Mtime
  0      0.463G       0        3.28        0       3.28      0.06663    11.43     3.069:   0%|    | 1/118286 [00:14<479:21:19, 14.59s/it]
  0      0.445G       1        3.491       0       3.49      0.03858    6.245     2.511:   0%|    | 2/118286 [00:17<255:55:56,  7.79s/it]
```

### Training Arguments
Core training arguments are listed below:
```
--dataset: the training dataset
--num_parts: the number of partitions
```

### Evaluation of AUC

### Evaluation of MN



## License
Copyright (c) 2023 xianfeng song. All rights reserved.
Licensed under the MIT License.
