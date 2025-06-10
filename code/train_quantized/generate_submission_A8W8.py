import gc, json, torch, logging, os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from llmcompressor import configure_logger, LoggerConfig

from evaluation_utils import evaluate_mmlu, mmlu_harness_hf, display_metric

configure_logger(LoggerConfig(
    disabled=True,
    clear_loggers=True,
    console_log_level=None,
    log_file=None,
    log_file_level=None
))

hub_prefix = "TheS3b/Qwen3-0.6B-SmoothQuant-W8A8-calib"  # base name for HF pushes
model_repo = "brygotti/MNLP_M3_mcqa_model"
BITS         = 8
BLOCK_SIZE   = 64
prompt_sizes = 200
MAX_SEQUENCE_LENGTH = 2048

logging.disable(logging.INFO)
os.environ["EXLLAMA_KERNELS_AVAILABLE"] = "0"

all_metrics = {}

# Tokenizer once
tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)

# Re-use the calibration set you already filtered
calibration_data = load_dataset("brygotti/MNLP_M3_mcqa_dataset")

def is_valid_prompt(example, min_len=64, max_len=256, thresh=0.5):
    tokens = tokenizer(example["prompt"], return_tensors="pt")["input_ids"]
    return min_len <= tokens.shape[1] <= max_len and (example["relevance_nlp4educ"] + example["relevance_mmlu"]) * 0.5 > thresh

filtered_calibration_set = calibration_data.filter(
    lambda ex: is_valid_prompt(ex), batched=False
).shuffle(seed=42)["train"]

def tokenise(sample):
    return tokenizer(
        sample["prompt"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )

eval_ds = load_dataset("brygotti/NLP4Education_english_single_mcq_4_choices")["test"]

for size in prompt_sizes:
    print(f"\n── SmoothQuant W{BITS}A8 with {size} calibration prompts ──\n")
    gc.collect()
    torch.cuda.empty_cache()

    # Build tokenised calibration dataset
    calib_ds = filtered_calibration_set.select(range(size)).map(
        tokenise, remove_columns=filtered_calibration_set.column_names
    )

    # FP16 baseline model
    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        torch_dtype=torch.float16,
        device_map="cuda",
        trust_remote_code=True,
    )

    recipe = [
        SmoothQuantModifier(
            smoothing_strength=0.8,
            ignore=["lm_head"],
            num_calibration_steps=size,
            block_size=BLOCK_SIZE,
        ),
        GPTQModifier(
            scheme=f"W{BITS}A8",
            targets="Linear",
            ignore=["lm_head"],
            block_size=BLOCK_SIZE,
        ),
    ]

    # One-shot quantisation pass
    oneshot(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=size,
    )

    model.eval()
    device = next(model.parameters()).device
    metrics = evaluate_mmlu(eval_ds, model, tokenizer, device, mmlu_harness_hf)
    display_metric(f"SmoothQuant W{BITS}A8 Size {size}", metrics)
    key = f"SmoothQuant W{BITS}A8 calib{size}"
    all_metrics[key] = metrics

    push_name = f"{hub_prefix}{size}"
    tokenizer.push_to_hub(push_name)
    model.push_to_hub(push_name)

    with open("Results/smooth_quant_metrics_calibration.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    del model
    gc.collect()
    torch.cuda.empty_cache()
