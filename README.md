# Deep CTS

## Description
This repository contains all the resources for my project "Boosting Performance in Deep Learning-based CTS Diagnosis" in Korea University Individual Study.

## Usage
1. Clone this repository:
```bash
$ git clone git@github.com:hijihyo/deep-cts.git
```

2. Create an environment and install necessary packages:
```bash
$ conda create -n deepcts python=3.9
$ conda init deepcts
$ pip install -r requirements.txt
```
3. Run the provided script file or modify it for your purpose:
```bash
$ ./scripts/ctsdiag_resnet.sh
# or
$ CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_tht_vit_attn --model tht_vit_attn --variant default --num-train-epochs 300 --per-device-train-batch-size 64
```
