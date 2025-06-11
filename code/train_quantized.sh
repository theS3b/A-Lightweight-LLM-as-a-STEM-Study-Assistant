#!/bin/bash

# PLEASE READ THE README, it explains way better everything than this script

echo "This script is designed to train quantized models using EfficientQAT, BNB-GPTQ, or SmoothQuant methods."
echo "It will install the necessary dependencies and set up the environment for training."
echo ""
echo "IT IS BETTER TO READ THE README.md FILE IN THE FOLDER train_quantized/ AND THE ONE IN train_quantized/EfficientQAT\ Adaptation/ FOR QAT, THIS SCRIPT IS JUST A CONVENIENCE."
echo ""
echo "Which quantization method do you want to train?"
echo "Options: EfficientQAT | BNB-GPTQ | SmoothQuant"
read -p "Enter your choice: " MODEL_TYPE
MODEL_TYPE=$(echo "$MODEL_TYPE" | tr '[:upper:]' '[:lower:]')

if [ "$MODEL_TYPE" == "efficientqat" ]; then
    echo "You are about to run the EfficientQAT Adaptation training script. Anaconda will be installed if not already present."
    echo "We recommend you to run this script on Izar, as it is optimized for that environment."
    read -p "Do you want to continue? (yes/no) " user_input
    user_input=$(echo "$user_input" | tr '[:upper:]' '[:lower:]')
    if [ "$user_input" != "yes" ]; then
        echo "Exiting the script."
        exit 0
    fi

    # Described in EfficientQAT Adaptation/README.md
    curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    bash Anaconda3-2024.10-1-Linux-x86_64.sh

    cd "EfficientQAT Adaptation/" || exit 1

    conda env create -f env.yml
    conda activate freshEffQAT

    pip install -v gptqmodel[triton] --no-build-isolation

    sbatch run_full_e2e.run
    sbatch run_full_ap4.run

    bash EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w2g64.sh
    bash EfficientQAT/examples/model_transfer/efficientqat_to_gptq/Qwen3-w4g64.sh

    echo ""
    echo "Do you want to evaluate the quantized models using lighteval?"
    echo "!!! Note: You MUST patch 'lighteval-epfl-mnlp' before evaluating. !!!"
    echo "Detailed instructions are provided in ----> EfficientQAT Adaptation/README.md <-----."
    read -p "Proceed with evaluation? (yes/no) " eval_input
    eval_input=$(echo "$eval_input" | tr '[:upper:]' '[:lower:]')

    if [ "$eval_input" == "yes" ]; then
        my_venvs_create lighteval_gptq
        my_venvs_activate lighteval_gptq
        pip install --upgrade accelerate optimum transformers
        pip install gptqmodel[triton] --no-build-isolation

        cd lighteval-epfl-mnlp/ || exit 1
        pip install -e .

        lighteval accelerate \
            --eval-mode "lighteval" \
            --save-details \
            --override-batch-size 4 \
            --custom-tasks "community_tasks/mnlp_mcqa_evals.py" \
            --output-dir "out/" \
            model_configs/quantized_model.yaml \
            "community|mnlp_mcqa_evals|0|0"
    else
        echo "Skipping model evaluation. You can run it later after patching lighteval-epfl-mnlp."
    fi
fi

if [ "$MODEL_TYPE" == "bnb-gptq" ]; then
    my_venvs_create sebm3_light_gptq
    my_venvs_activate sebm3_light_gptq
    pip install datasets transformers bitsandbytes accelerate torch tqdm optimum
    pip install gptqmodel --no-build-isolation

    echo "Please run the notebook Quantization Full Evaluation.ipynb to evaluate the models."
fi

if [ "$MODEL_TYPE" == "smoothquant" ]; then
    my_venvs_create sebm3_smooth_quant
    my_venvs_activate sebm3_smooth_quant
    pip install datasets transformers bitsandbytes accelerate torch tqdm optimum
    pip install llmcompressor

    echo "Please run the notebook Smooth Quant.ipynb to evaluate the models."
fi
