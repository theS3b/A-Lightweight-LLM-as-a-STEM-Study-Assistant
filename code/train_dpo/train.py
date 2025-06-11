from datasets import load_dataset
import torch
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import wandb

wandb.login()

SFT = False
beta = 0.10
epochs = 3
batch_size = 1
max_length = 1024

torch.cuda.empty_cache()

dataset_name = f"lindsaybordier/dpo_final_dataset_{max_length}"

if SFT:
    model_name = "brygotti/MNLP_M2_mcqa_model"
    save_name = f"Qwen3-0.6B-SFT-DPO_not-robust_argilla_acc4_beta{beta:.2f}"

else:
    model_name = "Qwen/Qwen3-0.6B-Base"
    save_name = f"Qwen3-0.6B-DPO_not-robust_final-dataset_acc4_beta{beta:.2f}"



model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
#model.gradient_checkpointing_enable()
#model = torch.compile(model)
tokenizer = AutoTokenizer.from_pretrained(model_name)
train_dataset = load_dataset(dataset_name, split="train")
eval_dataset = load_dataset(dataset_name, split="valid")

run = wandb.init(entity="lindsaybordier-epfl", name=save_name, project="MNLP_DPO_M2",
    #resume="allow",
    config={
        "architecture": "DPO",
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "trainer": "DPOTrainer",
        "model": model_name,
        "tokenizer": model_name,
        "max_prompt_length": max_length,
        "max_completion_length": max_length,
        "notes": f"beta={beta}, loss=default",
    }
)

training_args = DPOConfig(
    output_dir=f"checkpoints/{save_name}",
    #overwrite_output_dir=False,
    num_train_epochs=epochs,
    logging_steps=50,
    per_device_train_batch_size=batch_size,  # or smaller if you're testing on CPU
    per_device_eval_batch_size=batch_size,
    #max_prompt_length=max_length,
    #max_completion_length=max_length,
    max_length=max_length,
    report_to="wandb",
    save_strategy="epoch",
    eval_strategy="epoch",
    push_to_hub=True, # CHANGE TO AVOID OOM ON GNOTO
    hub_model_id=f"lindsaybordier/{save_name}",
    run_name=save_name,
    #loss_type="robust",
    optim="adamw_torch_fused",
    learning_rate=1e-5,
    gradient_accumulation_steps=4,
    weight_decay=0.0,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.01,
    beta=beta,
    metric_for_best_model="eval_loss",
    load_best_model_at_end=True,
)

trainer = DPOTrainer(model=model, 
                        args=training_args, 
                        processing_class=tokenizer, 
                        train_dataset=train_dataset, 
                        eval_dataset=eval_dataset)

torch.cuda.empty_cache()

trainer.train()
trainer.push_to_hub()

run.finish()