if [ ! -d train_dpo/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv train_dpo/.venv
    source train_dpo/.venv/bin/activate
    pip install -r train_dpo/requirements.txt
    echo "Virtual environment created and dependencies installed."
    echo "Running the training script..."
else
    echo "Virtual environment already exists. Activating it and running the training script..."
    source train_dpo/.venv/bin/activate
fi
python train_dpo/train.py