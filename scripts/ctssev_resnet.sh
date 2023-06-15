#!/bin/sh
# CTSSev ResNet (T)
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet --model onlyt_resnet --variant default --num-train-epochs 200 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet --model onlyt_resnet --variant default --num-train-epochs 200 --seed 43 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet --model onlyt_resnet --variant default --num-train-epochs 200 --seed 44 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet.out 2>&1;

# CTSSev ResNet (T, PT)
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet_pretrained --model onlyt_resnet --variant default --weights default --num-train-epochs 200 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet_pretrained.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet_pretrained --model onlyt_resnet --variant default --weights default --num-train-epochs 200 --seed 43 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet_pretrained.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet_pretrained --model onlyt_resnet --variant default --weights default --num-train-epochs 200 --seed 44 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet_pretrained.out 2>&1;

# CTSSev ResNet (T, L)
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet50 --model onlyt_resnet --variant resnet50 --num-train-epochs 200 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet50.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet50 --model onlyt_resnet --variant resnet50 --num-train-epochs 200 --seed 43 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet50.out 2>&1;
CUDA_VISIBLE_DEVICES=0 python src/dl_main.py --name ctssev_onlyt_resnet50 --model onlyt_resnet --variant resnet50 --num-train-epochs 200 --seed 44 --per-device-train-batch-size 64 > logs/ctssev_onlyt_resnet50.out 2>&1;

# CTSSev ResNet (THT, CC)
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctssev_tht_resnet_concat --model tht_resnet_concat --variant default --num-train-epochs 200 > logs/ctssev_tht_resnet_concat.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctssev_tht_resnet_concat --model tht_resnet_concat --variant default --num-train-epochs 200 --seed 43 > logs/ctssev_tht_resnet_concat.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctssev_tht_resnet_concat --model tht_resnet_concat --variant default --num-train-epochs 200 --seed 44 > logs/ctssev_tht_resnet_concat.out 2>&1;

# CTSSev ResNet (THT, Attn)
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctssev_tht_resnet_attn --model tht_resnet_attn --variant default --num-train-epochs 200 > logs/ctssev_tht_resnet_attn.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctssev_tht_resnet_attn --model tht_resnet_attn --variant default --num-train-epochs 200 --seed 43 > logs/ctssev_tht_resnet_attn.out 2>&1;
CUDA_VISIBLE_DEVICES=0,1 python src/dl_main.py --name ctssev_tht_resnet_attn --model tht_resnet_attn --variant default --num-train-epochs 200 --seed 44 > logs/ctssev_tht_resnet_attn.out 2>&1;
