# -*- coding: utf-8 -*-
"""train.ipynb

"""

from google.colab import drive
drive.mount('/content/drive')

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
from huggingface_hub import login

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/MyDrive/

# === CONFIGURATION ===
HF_TOKEN = "YOURHFTOKEN"
DATASET_NAME = "Alexhuou/embedder_train_doc"
OUTPUT_DIR = "MNLP_output"
PUSH_NAME = "Alexhuou/embedder_model_STxmmluV3"

# === Connexion Hugging Face
login(token=HF_TOKEN)

!pip install -U datasets huggingface_hub fsspec

from datasets import load_dataset
import random
import pandas as pd

# Load the test split of MMLU (only using for retrieval, no knowing whether right answer just subject)
mmlu = load_dataset("cais/mmlu", "all", split="test")

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(mmlu)

# Group by subject
subject_groups = df.groupby("subject")

# Create positive and negative pairs
data = []
for subject, group in subject_groups:
    questions = group["question"].tolist()
    if len(questions) < 2:
        continue
    for i in range(min(len(questions), 100)):  # limit samples per subject
        q = questions[i]
        # Pick a different sample in the same subject
        q_positive = random.choice([x for x in questions if x != q])

        # Pick a sample from a different subject
        other_subjects = [s for s in subject_groups.groups.keys() if s != subject]
        random_other = random.choice(other_subjects)
        q_negative = random.choice(subject_groups.get_group(random_other)["question"].tolist())

        data.append({
            "anchor": q,
            "positive": q_positive,
            "negative": q_negative,
            "subject": subject
        })

# Convert to DataFrame
triplet_df = pd.DataFrame(data)
triplet_df.head()

triplet_df.describe()

from datasets import Dataset

hf_dataset = Dataset.from_pandas(triplet_df)

# Define your repo name (replace with your actual HF username and desired repo name)
repo_name = DATASET_NAME

# Push to the Hub
hf_dataset.push_to_hub(repo_name)

# === Charger le dataset
dataset = load_dataset(DATASET_NAME)
train_data = dataset["train"]

# === Transformer en liste d'exemples pour TripletLoss
train_examples = [
    InputExample(texts=[ex["anchor"], ex["positive"], ex["negative"]])
    for ex in train_data
]

len(train_examples)

# === Charger le modèle SentenceTransformer
model_name = "thenlper/gte-small"

word_embedding_model = models.Transformer(model_name)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(),
    pooling_mode_mean_tokens=True
)

# Step 3: Combine into a SentenceTransformer model
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# === Préparer le DataLoader et la Loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
train_loss = losses.TripletLoss(model=model)

len(train_dataloader)

# === Entraînement
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=5,
    warmup_steps=10,
    output_path=OUTPUT_DIR
)

print(f"✅ Modèle entraîné sauvegardé dans : {OUTPUT_DIR}")

# === Push vers Hugging Face Hub
model.push_to_hub(PUSH_NAME)
print(f"📤 Modèle pushé sur Hugging Face : {PUSH_NAME}")