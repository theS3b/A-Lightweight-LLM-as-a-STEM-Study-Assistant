from datasets import load_dataset, Features, Value
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import numpy as np
import wandb
import os
import torch

###################### Evaluation code to reproduce lighteval pipeline ######################

LETTER_INDICES = ["A","B","C","D","E","F","G","H","I"]

class Doc:
    def __init__(self, query, choices, gold_index):
        self.query = query
        self.choices = choices
        self.gold_index = gold_index

def mmlu_harness_hf(line, multi_token=False):
    topic = "knowledge and skills in advanced master-level STEM courses"
    prompt = f"The following are multiple choice questions (with answers) about {topic.replace('_', ' ')}.\n\n"
    prompt += line["question"] + "\n"
    prompt += "".join([f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, line["choices"])])
    prompt += "Answer:"
    gold_idx = LETTER_INDICES.index(line["answer"])

    return Doc(
        query=prompt,
        choices = [f" {key}. {choice}" for key, choice in zip(LETTER_INDICES, line["choices"])] if multi_token else [" A", " B", " C", " D"],
        gold_index=gold_idx,
    )

@torch.no_grad()
def score_choice(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompt: str,
    choice: str
) -> float:
    """
    Returns the sum of log-probabilities that `model` assigns to `choice`
    when it is generated _after_ `prompt`.
    """

    # 1) Encode prompt alone (no special tokens, so we know its length exactly)
    enc_prompt = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    # 2) Encode the concatenation (prompt + choice) likewise
    enc_full = tokenizer(
        prompt + choice,
        return_tensors="pt",
        add_special_tokens=False
    ).to(device)

    input_ids = enc_full.input_ids        # shape (1, L_total)
    attn_mask = enc_full.attention_mask

    # 3) Forward pass to get logits and convert to log-probs
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    # 4) Sum the log-probs for exactly the choice tokens
    prompt_len = enc_prompt.input_ids.size(1)
    total_lp = 0.0

    # For each position i in the full sequence that corresponds to a choice token:
    #   the model’s probability for token j = input_ids[0, i] is in log_probs[0, i-1, j].
    for i in range(prompt_len, input_ids.size(1)):
        token_id = input_ids[0, i].item()
        total_lp += log_probs[0, i - 1, token_id].item()

    return total_lp

def evaluate_mmlu(
    dataset,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    harness_fn,
    multi_token=False
) -> float:
    """
    Run the MCQA evaluation over `dataset`, using `harness_fn` to convert each
    example into a Doc(query, choices, gold_index). Returns accuracy.
    """
    correct = 0
    total = 0

    for example in dataset:
        doc = harness_fn(example, multi_token)
        # score each choice
        scores = [
            score_choice(model, tokenizer, device, doc.query, c)
            for c in doc.choices
        ]
        
        pred = int(torch.argmax(torch.tensor(scores)))
        if pred == doc.gold_index:
            correct += 1
        total += 1

    return correct / total

###################### End of evaluation code ######################

###################### Main training code ######################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Qwen/Qwen3-0.6B-Base"
hf_user = "brygotti"
wandb_project = "MNLP-M2"
save_name = "MNLP_M3_mcqa_model"
cache_dir = None

# Fix issue with pytorch version
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.UInt32DType])

wandb.login()

run = wandb.init(project=wandb_project, name=save_name) #, id="???", resume="must")

data = load_dataset('brygotti/MNLP_M3_mcqa_dataset', split="train", cache_dir=cache_dir)
data = data.filter(lambda x: 1/2*(x['relevance_mmlu'] + x['relevance_nlp4educ']) > 0.42)
nlp4educ_data = load_dataset("brygotti/NLP4Education_english_single_mcq_4_choices", split="test", cache_dir=cache_dir)
mmlu_data = load_dataset("brygotti/mmlu", split="test", cache_dir=cache_dir)

print(f"Device set to {device}.")
print(f"Training on {len(data)} samples.")
print(f"Evaluating on nlp4educ with {len(nlp4educ_data)} samples.")
print(f"Evaluating on mmlu with {len(mmlu_data)} samples.")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

class CustomTrainer(SFTTrainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        old_acc_nlp4educ = evaluate_mmlu(nlp4educ_data, self.model, self.processing_class, device, mmlu_harness_hf)
        acc_nlp4educ = evaluate_mmlu(nlp4educ_data, self.model, self.processing_class, device, mmlu_harness_hf, multi_token=True)
        old_acc_mmlu = evaluate_mmlu(mmlu_data, self.model, self.processing_class, device, mmlu_harness_hf)
        acc_mmlu = evaluate_mmlu(mmlu_data, self.model, self.processing_class, device, mmlu_harness_hf, multi_token=True)
        metrics = {
            f"{metric_key_prefix}_nlp4educ_acc_old": old_acc_nlp4educ,
            f"{metric_key_prefix}_nlp4educ_acc": acc_nlp4educ,
            f"{metric_key_prefix}_mmlu_acc_old": old_acc_mmlu,
            f"{metric_key_prefix}_mmlu_acc": acc_mmlu,
        }
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

training_args = SFTConfig(
    output_dir=f"checkpoints/{save_name}",
    overwrite_output_dir=False,

    completion_only_loss=True,
    max_length=1024,
    per_device_train_batch_size=2,
    num_train_epochs=1,

    report_to="wandb",
    logging_steps=50,
    run_name=save_name,

    save_strategy="epoch",
    push_to_hub=True,
    hub_model_id=f"{hf_user}/{save_name}",

    # per_device_eval_batch_size=2, # Not used anyway given our custom evaluation fn
    eval_strategy="epoch"
    # eval_on_start=True,
)

trainer = CustomTrainer(
    model,
    processing_class=tokenizer,
    train_dataset=data.select_columns(['prompt', 'completion']),
    # This is required for the trainer to run evaluation, although we ignore the dataset in our custom evaluation fn, we still need to pass it here. Since it should be in the same format as train_dataset, I figured I could just pass train_dataset here too. It will be ignored anyways
    eval_dataset=data.select_columns(['prompt', 'completion']), 
    args=training_args,
)

trainer.train() # resume_from_checkpoint=True)

run.finish()
