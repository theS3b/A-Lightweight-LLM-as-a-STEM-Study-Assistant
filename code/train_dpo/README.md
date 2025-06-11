# DPO model training code

## Running the bash script
To run the bash script for training the DPO model, you need to have Python 3.13.2 installed on your system. The script will automatically create a virtual environment and install all the necessary dependencies.

## Preprocessing the data
The directory `data_preprocessing` contains the code for preprocessing the data. More specifically:
- `generate_train_data.ipynb`: This Jupyter notebook generates the training data for the MCQA model and pushes it to the Huggingface Hub.
- `generate_relevance_data.ipynb`: This Jupyter notebook generates the relevance columns for the training data and pushes it to the Huggingface Hub.
- `analyze_relevance.ipynb`: This Jupyter notebook contains plots and analysis of the relevance columns generated in the previous notebook.
- `upload_nlp4education.ipynb`, `upload_mmlu.ipynb`: These Jupyter notebooks upload the filtered NLP4Education and MMLU, formatted for the evaluation pipeline, to the Huggingface Hub. Thoses 2 datasets were used for evaluating the MCQA model.

## Training the model
The training code is located in the `train.py` file. Make sure to set the correct Huggingface Hub username so that the model can be pushed to your account. You will also need to be logged in to WandB to log the training process.
