# EfficientQAT for Qwen3

This folder contains our modifications to EfficientQAT and scripts to train, evaluate, and convert QAT checkpoints to GPTQ checkpoints.

## Adaptations

- We found a viable configuration for the environment that works with Qwen-3 models and is compatible with the original triton version of EfficientQAT. It is defined in `env.yml`.
- We modified [efficientqat_to_others.py](EfficientQAT/model_transfer/efficientqat_to_others.py) to support conversion to TORCH formats. We also did small changes as the code used an older version of the `gptqmodel` library. Lastly, we had to add code to manually add the quantization configuration to the `config.json` of the converted model as the original code does not handle this via parameters of the `from_quantized` method in the newer version of GPTQModel.

- We modified [datautils_block.py](EfficientQAT/datautils_block.py) to handle the new data format used in EfficientQAT. We decided to cut the input sequence randomly, but before the label token. We could experiment with other strategies like cutting at the label token.

- In [main_e2e_qp.py](EfficientQAT/main_e2e_qp.py), we changed from using Seq2SeqTrainer to Trainer. We also added a call to our custom [evaluation_utils.py](EfficientQAT/evaluation_utils.py) to evaluate the model after training.

## Environment setup on Izar

Install Anaconda (if needed) and create the `freshEffQAT` environment:

```bash
# Install Anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh

# Create & activate the QAT env
conda env create -f env.yml
conda activate freshEffQAT

# Install gptqmodel with Triton support
pip install -v gptqmodel[triton] --no-build-isolation
```

## Training with EfficientQAT

Please refer to the following slurm scripts for training QAT models on Qwen-3:
- [run_full_ap4.run](run_full_ap4.run): Block-AP & E2E-QP training for Qwen-3 4-bit model.
- [run_full_ap2.run](run_full_ap2.run): Block-AP & E2E-QP training for Qwen-3 2-bit model.

Then you can run the following scripts to convert the QAT checkpoints to GPTQ format (do it on GPU as well, this part is very fast though):
- [EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w2g64.sh](EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w2g64.sh)
- [EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w4g64.sh](EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w4g64.sh)

Finally, you can push the converted models to the Hugging Face Hub using the following notebook:
- [push_to_hub.ipynb](push_to_hub.ipynb)

### 1. Block‐wise Approximate Pretraining (Block-AP)

Train each transformer block independently: see [examples/block_ap/w4g64.sh](EfficientQAT/examples/block_ap/Qwen3/w4g64.sh) for an example script.

### 2. End-to-End Quantization Parameter Tuning (E2E-QP)

Refine quantization parameters jointly: see [examples/e2e_qp/w4g64.sh](EfficientQAT/examples/e2e_qp/Qwen3/w4g64.sh) for an example script.

### 3. Converting QAT Checkpoints to GPTQ Format

Once you have a QAT checkpoint, convert it for GPTQ inference: see [examples/model_transfer/efficientqat_to_gptq/convert.sh](EfficientQAT%20Adaptation/EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w4g64.sh) for an example script.


## Folder structure

```
.
├── env.yml
├── logs/                    # training and eval logs
│   
├── EfficientQAT/            # QAT code
│   ├── main_block_ap.py
│   ├── main_e2e_qp.py
│   ├── evaluation_utils.py
│   ├── datautils_e2e.py
│   └── model_transfer/
│       └── efficientqat_to_others.py
└── examples/                # launch scripts & sample configs
    ├── block_ap/
    │   └── run_block_ap.sh
    ├── e2e_qp/
    │   └── run_e2e_qp.sh
    └── model_transfer/
        └── run_model_transfer.sh
```
