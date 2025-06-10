# EfficientQAT for Qwen3

This folder contains our modifications to EfficientQAT and scripts to train, evaluate, and convert QAT checkpoints to GPTQ checkpoints.

## Adaptations

* **env.yml** defines a compatible conda environment for Qwen-3 and Triton-based `gptqmodel`.
* **model\_transfer/efficientqat\_to\_others.py** supports conversion to Torch/GPTQ formats, patches new `gptqmodel` APIs, and injects quantization settings into the model's `config.json`.
* **datautils\_block.py** cuts input sequences randomly (before the label token) to fit block-wise QAT.
* **main\_e2e\_qp.py** uses `Trainer` (instead of `Seq2SeqTrainer`) and hooks in `evaluation_utils.py` for validation during training.

## Environment setup on Izar

Install Anaconda and create the `freshEffQAT` environment:

```bash
# Install Anaconda (if needed)
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh

# Create & activate the QAT env
conda env create -f env.yml
conda activate freshEffQAT

# Install Triton-enabled GPTQModel
pip install -v gptqmodel[triton] --no-build-isolation
```

## Training with EfficientQAT

Use the provided Slurm scripts:

* **run\_full\_ap4.run** — Block-AP & E2E-QP for 4-bit Qwen-3
* **run\_full\_ap2.run** — Block-AP & E2E-QP for 2-bit Qwen-3

Convert QAT checkpoints to GPTQ (GPU required, fast):

```bash
bash EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w4g64.sh
bash EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w2g64.sh
```

Push converted models with \[push\_to\_hub.ipynb].

### 1. Block‐AP

See `examples/block_ap/Qwen3/w4g64.sh`.

### 2. E2E-QP

See `examples/e2e_qp/Qwen3/w4g64.sh`.

### 3. GPTQ Conversion

See `model_transfer/efficientqat_to_others.py` and `examples/model_transfer/efficientqat_to_gptq/*.sh`.

## How to evaluate with Light-Eval (from EPFL)

Follow these steps to run your Qwen3 QAT model under the lighteval framework:

1. **Patch the code**

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

   ### If it really doesn't work
   First, try the above, if it really doesn't work, you can try the following:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the model**

   ```yaml
   # quantized_model.yaml
   model:
     base_params:
       model_args: "pretrained=TheS3b/Qwen3-EfficientQAT-w4g64,revision=main,trust_remote_code=True"
       use_chat_template: false
       dtype: "auto"
       load_in_optimum: false
       compile: false
   ```

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

## Acknowledgements
This work builds upon the EfficientQAT framework and integrates it with the Qwen-3 model. We thank the original authors for their contributions and the community for ongoing support. 