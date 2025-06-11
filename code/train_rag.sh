if [ ! -d train_rag/.venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv train_rag/.venv
    source train_rag/.venv/bin/activate
    echo "Virtual environment created"
    echo "Running the training script..."
else
    echo "Virtual environment already exists. Activating it and running the training script..."
    source train_rag/.venv/bin/activate
fi
python train_rag/train.py
