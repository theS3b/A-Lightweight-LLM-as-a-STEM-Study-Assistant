import time
from typing import Dict, List
from tqdm import tqdm
from datasets import load_dataset
import torch

LETTER_INDICES: List[str] = ["A", "B", "C", "D"]

class Doc:
    def __init__(self, query: str, choices: List[str], gold_index: int):
        self.query = query
        self.choices = choices
        self.gold_index = gold_index

def mmlu_harness_hf(
    ex, topic: str = "knowledge and skills in advanced master-level STEM courses"
) -> Doc:
    """
    Convert a raw example from the HF MMLU JSON into a Doc object
    understood by the evaluator.
    """
    question = ex["question"]
    choices = ex["choices"]
    answer = ex["answer"]
    prompt = f"The following are multiple choice questions about {topic}.\n\n"
    prompt += question + "\n"
    for letter, text in zip(LETTER_INDICES, choices):
        prompt += f"{letter}. {text}\n"
    prompt += "Answer:"
    gold_ix = LETTER_INDICES.index(answer)
    # prepend a space before each candidate, as required by the original prompt logic
    return Doc(prompt, [f" {c}" for c in LETTER_INDICES], gold_ix)

@torch.no_grad()
def score_choice(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    prompt: str,
    choice: str,
) -> float:
    """
    Log-probability that `model` assigns to `choice` when it is generated
    directly after `prompt`.
    """
    # 1) Encode prompt and prompt+choice (no special tokens)
    enc_prompt = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)
    enc_full = tokenizer(
        prompt + choice, return_tensors="pt", add_special_tokens=False
    ).to(device)

    input_ids = enc_full.input_ids
    attn_mask = enc_full.attention_mask

    # 2) Forward pass
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    log_probs = torch.log_softmax(logits, dim=-1)

    # 3) Sum log-probs only for the choice tokens
    prompt_len = enc_prompt.input_ids.size(1)
    total_lp = 0.0
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
) -> Dict[str, float]:

    correct = total = 0
    total_time = 0.0
    total_tokens = 0
    per_example_peaks: List[int] = []

    for example in tqdm(dataset, desc="Evaluating"):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()

        doc = harness_fn(example)

        for choice in doc.choices:
            # same construction we pass to the model
            ids = tokenizer(doc.query + choice,
                            add_special_tokens=False).input_ids
            total_tokens += len(ids)

        scores = [
            score_choice(model, tokenizer, device, doc.query, c)
            for c in doc.choices
        ]

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        total_time += time.perf_counter() - start

        if device.type == "cuda":
            per_example_peaks.append(torch.cuda.max_memory_allocated(device))

        pred = int(torch.argmax(torch.tensor(scores)))
        correct += (pred == doc.gold_index)
        total += 1

    avg_time      = total_time / total
    tokens_per_s  = total_tokens / total_time
    avg_peak_vram = (
        (sum(per_example_peaks) / len(per_example_peaks)) / 1024**2
        if per_example_peaks else float("nan")
    )

    return {
        "accuracy": correct / total,
        "avg_time_s": avg_time,
        "tokens_per_s": tokens_per_s,
        "avg_peak_vram_MB": avg_peak_vram,
        "score_acc_over_vram": 1000 * (correct / total) / avg_peak_vram
    }


def display_metric(name, metrics):
    print(
        f"\n**{name} Evaluation Results**\n"
        f"- Accuracy              : {metrics['accuracy'] * 100:6.2f} %\n"
        f"- Avg. inference time   : {metrics['avg_time_s'] * 1_000:6.1f} ms\n"
        f"- Throughput (tok/s)    : {metrics['tokens_per_s']:6.1f}\n"
        f"- Avg. peak VRAM        : {metrics['avg_peak_vram_MB']:6.1f} MB\n"
        f"- Score Acc/VRAM        : {metrics['score_acc_over_vram']:6.3f} \n"
    )
