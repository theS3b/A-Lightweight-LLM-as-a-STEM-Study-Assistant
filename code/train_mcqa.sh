if [ ! -d train_mcqa/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv train_mcqa/.venv
    source train_mcqa/.venv/bin/activate
    pip install -r train_mcqa/requirements.txt
    echo "Virtual environment created and dependencies installed."
    echo "Running the training script..."
else
    echo "Virtual environment already exists. Activating it and running the training script..."
    source train_mcqa/.venv/bin/activate
fi
python train_mcqa/train.py

