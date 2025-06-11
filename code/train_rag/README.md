# MNLP Project - RAG for Multi-Choice QA

This submission contains the components developed for the RAG-based Multi-Choice QA system designed as part of the EPFL MNLP project. The goal was to build a system that retrieves relevant context from a constructed corpus and answers MCQA-style questions effectively.

---

## Overview of Submitted Files

### ✅ `train.ipynb`
Finetunes a sentence embedding model using triplet data (anchor, positive, negative). While the resulting finetuned model was functional, its accuracy was lower than the pretrained `gte-large`, which remained the best-performing embedder. Nonetheless, the training notebook is included to demonstrate the methodology.

### ✅ `corpus_maker.ipynb`
Builds a custom corpus for the RAG component:
- Multiple datasets were sampled from HuggingFace.
- Prompts were sent to OpenAI's GPT to generate similar question variants.
- Resulting rows consist of a generated `text` and a `source` field.

**Important Note**: Much of the corpus building and dataset exploration was done manually in an iterative manner. This included testing and evaluating various dataset combinations and manually inspecting results. These intermediate steps were not saved but were essential to achieving the final version of the corpus.

### ✅ `Copy_of_MNLP_EPFL_lighteval_V2.ipynb`
Notebook adapted for RAG evaluations on Google Colab:
- Integrates a retriever powered by a FAISS index built from the generated corpus.
- Allows reuse of an existing FAISS index, avoiding redundant computation.

---

## Additional Artifacts

### 📂 `MNLP_output/`
Contains all generated results from the LightEval evaluation runs. This includes detailed output files with retrieval and prediction information.

### 📂 `lighteval/`
A modified version of the original LightEval framework, adapted to:
- Enable RAG-style retrievals.
- Support custom FAISS index reuse for faster evaluation.

---

## Remarks

- The RAG approach required substantial experimentation with datasets, prompts, and retriever-generator combinations.
- Dataset wrangling was largely done interactively (i.e., exploring in cells and adjusting scripts based on observed outputs).
- While the custom-trained embedder underperformed compared to `gte-large`, it is included for completeness.

---

## How to Use

You can simulate embedding training and corpus building along with rag evaluation using:

```bash
# Training (optional)
python train.py

# Corpus generation
python corpus_maker.py

# RAG evaluation
python Copy_of_MNLP_EPFL_lighteval_V2.py
```

---

## Dependencies

The notebooks already contain the pip install commands needed to run the code.
