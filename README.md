# 2nd-Order-Opt

# ResNet-50-NG+ Example

## Description

This is an example of training ResNet-50 V1.5 with ImageNet2012 dataset by second-order optimizer NG+. NG+ is a novel approximate seond-order optimization method. With fewer iterations, SENG can finish ResNet-50 V1.5 training within 40 epochs to top-1 accuracy of 75.9% using 4 Tesla V100 GPUs, which is much faster than SGD with Momentum.

## Model Architecture

The overall network architecture of ResNet-50 is show below:[link](https://arxiv.org/pdf/1512.03385.pdf)

## Dataset

Dataset used: ImageNet2012

- Dataset size 224*224 colorful images in 1000 classes
    - Train：1,281,167 images  
    - Test： 50,000 images
- Data format：jpeg
    - Note：Data will be processed in dataset.py

- Download the dataset ImageNet2012

> Unzip the ImageNet2012 dataset to any path you want and the folder structure should include train and eval dataset as follows:

```shell
    ├── train           # train dataset
    └── val             # infer dataset
```

## Environment Requirements

- Hardware（GPU）
    - Prepare hardware environment with GPU processors.
- Framework
    - pytorch (>= 1.5)
    - tensorboardX
    - [NVIDIA DALI](https://developer.nvidia.com/dali)

## Quick Start

```python
python -m torch.distributed.launch --master_port 12111 --nproc_per_node=4 main_ngplus.py --fp16   --logdir your_log_file  --lr-decay-rate 0.8 --lr_exponent=5.0 --epoch_end 52 --curvature_momentum 0.99 --damping 0.16 --lr_init 0.18  --warmup_epoch 5 --batch_size 64 --datadir your_data_dir
```

### Code Structure

- NGPlus.py: Our NG+ optimizer.
- dali_pipe.py: A wrapper over DALI.
- data_manager.py: Utilities about loading datasets.
- logging_utils.py: Utilities about logging.
- main_ngplus.py: The main script for training.
- nvidia_dali_utils2.py: Utilities about DALI.
- resnet.py: The definition of resnet50 model.
- utils.py: Miscellaneous utilities.