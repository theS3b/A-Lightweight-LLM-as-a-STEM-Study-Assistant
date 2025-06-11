CUDA_VISIBLE_DEVICES=0 python main_block_ap.py \
--model brygotti/MNLP_M2_mcqa_model  \
--output_dir /scratch/izar/delsad/mnlp_cache/EfficientQAT/block_ap_log/Qwen3-w4g64 \
--wbits 4 \
--calib_dataset get_mnlp_randomsampling \
--cache_dir /scratch/izar/delsad/mnlp_cache/EfficientQAT/cache \
--group_size 64 \
--quant_lr 1e-4 \
--weight_lr 2e-5 \
--training_seqlen 128 \
--real_quant \
# --eval_ppl \
# --eval_tasks arc_easy,medmcqa \
--save_quant_dir /scratch/izar/delsad/mnlp_cache/EfficientQAT/block_ap_models/Qwen3-w4g64