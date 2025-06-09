#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python  -m model_transfer.efficientqat_to_others \
--model /scratch/izar/delsad/mnlp_cache/EfficientQAT/e2e-qp-output/Qwen3-w2g64/checkpoint-1000/ \
--save_dir /scratch/izar/delsad/mnlp_cache/EfficientQAT/gptq/Qwen3-w2g64 \
--wbits 2 \
--group_size 64 \
--target_format gptq