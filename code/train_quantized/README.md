# Quantization Experiments

This directory contains Jupyter notebooks and scripts to train, evaluate, and push quantized Qwen-3 models for multiple-choice QA (MCQA) on the NLP4Education benchmark.

## Main Notebooks

- **Quantization Full Evaluation.ipynb**  
  Baselines and post-training quantization (PTQ):
  - **Base & SFT models**: load full-precision checkpoints and measure MCQA accuracy.
  - **BitsAndBytes PTQ**: 4-bit (`bnb_4bit`) and 8-bit (`int8`) weight-only quantization via `BitsAndBytesConfig`.
  - **GPTQ**: 4-bit quantization with varying calibration sizes (20, 200, 2000 prompts) via `GPTQConfig`.
  - Pushes each quantized model & tokenizer to Hugging Face Hub.

- **Smooth Quant.ipynb**  
  A variant of PTQ combining SmoothQuant (rescale weights and activations) with GPTQ.  
  Same evaluation pipeline as above.

- **QAT Evaluation.ipynb**  
  Quantization-aware training (QAT) via EfficientQAT:
  - Load 4-bit (`w4g64`) and 2-bit (`w2g64`) pretrained QAT checkpoints (`TheS3b/Qwen3-EfficientQAT-…`).
  - Evaluate on test split with `evaluate_mmlu()` and record metrics to `Results/QAT-metrics.json`.

- **QAT Training**  
  The EfficientQAT adaptation is documented in [EfficientQAT Adaptation/README.md](EfficientQAT%20Adaptation/README.md).
  It includes details on how to train QAT models, including the training script and configuration.

## Requirements

### Quantization Full Evaluation.ipynb & QAT Evaluation.ipynb
```bash
my_venvs_create sebm3_light_gptq
my_venvs_activate sebm3_light_gptq
pip install datasets transformers bitsandbytes accelerate torch tqdm optimum
pip install gptqmodel --no-build-isolation
```

### Smooth Quant.ipynb
```bash
my_venvs_create sebm3_smooth_quant
my_venvs_activate sebm3_smooth_quant
pip install datasets transformers bitsandbytes accelerate torch tqdm optimum
pip install llmcompressor
```


## Usage

1. **Run notebooks**
   Launch each notebook in order to reproduce results:

   * `Quantization Full Evaluation.ipynb`
   * `Smooth Quant.ipynb`
   * `QAT Evaluation.ipynb`

2. **Results**

   * Metrics are logged in `Results/`.
   * Results are also shown in the notebook outputs.

## Directory Structure

```
.
├── Quantization Full Evaluation.ipynb
├── Smooth Quant.ipynb
├── QAT Evaluation.ipynb
├── evaluation_utils.py  # Replicates core evaluation of lighteval
└── Results/
    ├── quantization_metrics.json
    └── QAT-metrics.json
└── EfficientQAT Adaptation/
    └── README.md  # Contains all details about EfficientQAT adaptation
```