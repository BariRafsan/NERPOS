import os
import config
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


def train_test_split_func(X, y_pos, y_ner):
    X_train, X_test, y_pos_train, y_pos_test, y_ner_train, y_ner_test = (
        train_test_split(X, y_pos, y_ner, test_size=0.2, random_state=42)
    )
    return X_train, X_test, y_pos_train, y_pos_test, y_ner_train, y_ner_test


def load_latest_model(model_dir):
    print(f"Loading model from: {model_dir}")
    model_files = [
        f for f in os.listdir(model_dir) if f.endswith(".keras")
    ]  

    model_files.sort(reverse=True)  
    if not model_files:
        raise FileNotFoundError("No model files found in the specified directory.")

    # Get the latest model file
    latest_model_file = model_files[0]
    latest_model_path = os.path.join(model_dir, latest_model_file)
    print(f"Loading model: {latest_model_path}")
    # Load the model
    model = load_model(latest_model_path)
    print(f"Loaded model: {latest_model_file}")
    return model
