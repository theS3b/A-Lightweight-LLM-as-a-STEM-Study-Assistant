#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python main_e2e_qp.py \
    --quant_model_path /scratch/izar/delsad/mnlp_cache/EfficientQAT/block_ap_models/Qwen3-w3g64 \
    --wbits 3 \
    --group_size 64 \
    --learning_rate 1e-5 \
    --dataset mnlp \
    --dataset_format mnlp \
    --output_dir /scratch/izar/delsad/mnlp_cache/EfficientQAT/e2e-qp-output/Qwen3-w3g64 \
    --do_train True \
    --do_mmlu_eval False \
    --source_max_len 384 \
    --target_max_len 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --save_strategy steps \
    --evaluation_strategy steps \
    --max_steps 1000 \
    --eval_steps 2000 \
    --eval_dataset_size 16 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --group_by_length \
    --run_custom_test True \