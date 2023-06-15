#!/bin/sh
# CTSDiag ResNet (T)
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet --model onlyt_resnet --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet --model onlyt_resnet --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 43 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet --model onlyt_resnet --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 44 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet.out 2>&1;

# CTSDiag ResNet (T, PT)
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet_pretrained --model onlyt_resnet --variant default --weights default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet_pretrained.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet_pretrained --model onlyt_resnet --variant default --weights default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 43 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet_pretrained.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet_pretrained --model onlyt_resnet --variant default --weights default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 44 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet_pretrained.out 2>&1;

# CTSDiag ResNet (T, L)
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet50 --model onlyt_resnet --variant resnet50 --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet50.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet50 --model onlyt_resnet --variant resnet50 --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 43 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet50.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctsdiag_onlyt_resnet50 --model onlyt_resnet --variant resnet50 --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 44 --per-device-train-batch-size 64 > logs/ctsdiag_onlyt_resnet50.out 2>&1;

# CTSDiag ResNet (THT, CC)
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctsdiag_tht_resnet_concat --model tht_resnet_concat --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 > logs/ctsdiag_tht_resnet_concat.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctsdiag_tht_resnet_concat --model tht_resnet_concat --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 43 > logs/ctsdiag_tht_resnet_concat.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctsdiag_tht_resnet_concat --model tht_resnet_concat --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 44 > logs/ctsdiag_tht_resnet_concat.out 2>&1;

# CTSDiag ResNet (THT, Attn)
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctsdiag_tht_resnet_attn --model tht_resnet_attn --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 > logs/ctsdiag_tht_resnet_attn.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctsdiag_tht_resnet_attn --model tht_resnet_attn --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 43 > logs/ctsdiag_tht_resnet_attn.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctsdiag_tht_resnet_attn --model tht_resnet_attn --variant default --num-train-epochs 200 --dataset ctsdiag --num-classes 2 --seed 44 > logs/ctsdiag_tht_resnet_attn.out 2>&1;
