import os
from . import config
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from .data_preprocessing import load_processed_data

X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder = load_processed_data(
    config.processed_data_dir
)
X_train, X_test, y_pos_train, y_pos_test, y_ner_train, y_ner_test = train_test_split(
    X, y_pos, y_ner, test_size=0.2, random_state=42
)


def load_latest_model(model_dir):
    # Get a list of all model files in the directory
    model_files = [f for f in os.listdir(model_dir) if f.endswith(".keras")]

    # Sort the files by their timestamp (assuming the format "my_model_YYYYMMDD-HHMMSS.h5")
    model_files.sort(reverse=True)  # Sort in descending order so the latest is first

    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")

    # Get the latest model file
    latest_model_file = model_files[0]
    latest_model_path = os.path.join(model_dir, latest_model_file)

    # Load the model
    model = load_model(latest_model_path)
    print(f"Loaded model: {latest_model_file}")
    return model


model = load_latest_model(config.model_dir)
