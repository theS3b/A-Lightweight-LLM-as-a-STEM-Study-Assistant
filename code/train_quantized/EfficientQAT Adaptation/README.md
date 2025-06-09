# EfficientQAT

## EfficientQAT environment setup on Izar

Install Anaconda if not already installed:
```bash
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

Create a new conda environment:
```bash
conda env create -f env.yml
conda activate freshEffQAT

# Install gptqmodel with triton support
pip install -v gptqmodel[triton] --no-build-isolation
```

## Running the EfficientQAT example
