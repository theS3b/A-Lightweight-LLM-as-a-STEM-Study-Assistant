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

## How to evaluate GPTQModel (EfficientQAT & GPTQ) with Light-Eval (from EPFL)

#### Please see EfficientQAT Adaptation/README.md for more details including how to train QAT models.

Follow these steps to run your Qwen3 QAT model under the lighteval framework:

1. **Patch lighteval**

   ```diff
   --- transformers_model.py.orig
   +++ transformers_model.py
   @@ -50,7 +50,6 @@
   - from optimum.quanto import QuantizedModelForCausalLM  # remove this
   +
   @@ -564,6 +564,9 @@
        if self.load_in_optimum:
   +        from optimum.quanto import QuantizedModelForCausalLM  # add back here
   +        model = QuantizedModelForCausalLM.from_pretrained(
        
   @@ -246,7 +247,7 @@
            if model_auto_quantization_config["quant_method"] == "gptq":
   -            # if not is_autogptq_available():
   -            #     raise ImportError(NO_AUTOGPTQ_ERROR_MSG)
   +            # commented out legacy GPTQ check
             auto_config.quantization_config["use_exllama"] = None
             self.quantization_config = GPTQConfig(
                 **auto_config.quantization_config,
   ```

2. **Create & activate environment**

   ```bash
   my_venvs_create lighteval_gptq
   my_venvs_activate lighteval_gptq
   pip install --upgrade accelerate optimum transformers
   pip install gptqmodel[triton] --no-build-isolation
   # ignore any resolver warnings

   cd lighteval-epfl-mnlp/
   pip install -e .   # do NOT add [quantization]
   ```

   ### If the above really doesn't work
   First, try the above, if it really doesn't work, you can try the following:
   ```bash
   my_venvs_create lighteval_gptq_hope
   my_venvs_activate lighteval_gptq_hope
   pip install -r requirements.txt
   ```

## Other Requirements for Training all but QAT
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
