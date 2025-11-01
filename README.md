# A Lightweight LLM as a STEM Study Assistant

<font size=4><div align='center'>[[📄 Tech Report](pdf/A%20Lightweight%20LLM%20as%20a%20STEM%20Study%20Assistant.pdf)]</div></font>

### 📝 Abstract

*We present a domain-adapted LLM for answering STEM multiple-choice questions. We explore fine-tuning Qwen3-0.6B-Base using SFT, Direct Preference Optimization (DPO), retrieval-augmented generation (RAG), quantization, and dataset filtering. Combining alignment, retrieval, and efficiency techniques, our approach improves MCQA accuracy while enabling resource-efficient deployment.*


## 🛠️ Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- Hugging Face account (for model and dataset access)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/theS3b/A-Lightweight-LLM-as-a-STEM-Study-Assistant.git
cd A-Lightweight-LLM-as-a-STEM-Study-Assistant
```

2. Install dependencies for each training module (navigate to the respective directories):
```bash
# DPO training
cd code/train_dpo
pip install -r requirements.txt

# MCQA training
cd ../train_mcqa
pip install -r requirements.txt

# Quantized model training (see README in code/train_quantized, minimal installation as follows)
cd ../train_quantized
pip install -r requirements.txt

# RAG training
cd ../train_rag
pip install -r requirements.txt
```


## 🚀 Training

We provide four training pipelines for different model configurations. Each can be run independently using the provided bash scripts.

### 1. DPO Model Training
Train a model using Direct Preference Optimization to align with educational preferences:
```bash
bash code/train_dpo.sh
```

Detailed implementation and preprocessing steps can be found in `code/train_dpo/`.

### 2. MCQA Model Training
Fine-tune a model for multiple-choice question answering tasks:
```bash
bash code/train_mcqa.sh
```

Preprocessing notebooks for dataset generation and relevance analysis are available in `code/train_mcqa/preprocessing/`.

### 3. Quantized Model Training
Train and quantize models for efficient deployment:
```bash
bash code/train_quantized.sh
```

This includes Quantization-Aware Training (QAT) and Smooth Quantization techniques. Evaluation notebooks and detailed results are in `code/train_quantized/`.

### 4. RAG Model Training
Train a model with Retrieval-Augmented Generation capabilities:
```bash
bash code/train_rag.sh
```

Corpus creation and training notebooks are provided in `code/train_rag/`.


## 📊 Evaluation

We provide comprehensive evaluation tools to benchmark model performance on STEM tasks. The evaluation suite assesses:
- MCQA accuracy on MMLU and NLP4Education benchmarks
- DPO preference alignment
- Quantization impact on model quality
- RAG retrieval precision and answer quality

Evaluation results and metrics can be found in `code/train_quantized/Results/`.


## 📦 Model Checkpoints

All trained models are available on Hugging Face Hub:

- **DPO Model**: [lindsaybordier/MNLP_M3_dpo_model](https://huggingface.co/lindsaybordier/MNLP_M3_dpo_model)
- **MCQA Model**: [brygotti/MNLP_M3_mcqa_model](https://huggingface.co/brygotti/MNLP_M3_mcqa_model)
- **Quantized Model**: [TheS3b/MNLP_M3_quantized_model](https://huggingface.co/TheS3b/MNLP_M3_quantized_model)
- **RAG Model**: [Alexhuou/MNLP_M3_rag_model](https://huggingface.co/Alexhuou/MNLP_M3_rag_model)
  - Document Encoder: [Alexhuou/MNLP_M3_document_encoder](https://huggingface.co/Alexhuou/MNLP_M3_document_encoder)
  - RAG Documents: [Alexhuou/merged_rag_docs_final](https://huggingface.co/datasets/Alexhuou/merged_rag_docs_final)

Model configurations are provided in the `model_configs/` directory. Training datasets are also available through the references in `data/data_repo.json`.


## 📁 Repository Structure

```
├── code/                          # Training code and scripts
│   ├── train_dpo/                # DPO training implementation
│   ├── train_mcqa/               # MCQA training and preprocessing
│   ├── train_quantized/          # Quantization experiments
│   └── train_rag/                # RAG training and corpus creation
├── model_configs/                # Model configuration files
├── data/                         # Dataset references
├── pdf/                          # Technical report
└── _test/                        # Validation scripts
```


## 🤝 Acknowledgements

This project was developed as part of the Modern Natural Language Processing (CS-552) course at EPFL. We acknowledge the instructors and TAs for their efforts in running the course. We also acknowledge the [lighteval-epfl-mnlp](https://github.com/eric11eca/lighteval-epfl-mnlp) evaluation suite, on which our customized benchmarking implementation is based. We also acknowledge the use of evaluation suite for model benchmarking.


## 📄 Citation

If you find this work useful, please cite our technical report:
```bibtex
@techreport{stem-study-assistant-2025,
  title={A Lightweight LLM as a STEM Study Assistant},
  author={Sébastien Delsad, Bryan Gotti, Lindsay Bordier, Alexandre Huou},
  year={2025},
  institution={EPFL}
}
```
