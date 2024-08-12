import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np


def load_data(filepath):
    """Loads the dataset from a file."""
    data = []
    with open(filepath, "r") as file:
        sentence = []
        for line in file:
            if line.strip():
                if "\t" not in line:
                    if sentence:
                        data.append(sentence)
                    sentence = []
                else:
                    try:
                        token, pos_tag, ner_tag = line.strip().split("\t")
                        sentence.append((token, pos_tag, ner_tag))
                    except Exception:
                        continue
        if sentence:
            data.append(sentence)
    return data


def preprocess_data(data, max_length=50):
    """Preprocess the dataset: tokenize, encode, and pad sequences."""
    sentences = [[token for token, _, _ in s] for s in data]
    pos_tags = [[pos for _, pos, _ in s] for s in data]
    ner_tags = [[ner for _, _, ner in s] for s in data]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)

    pos_encoder = LabelEncoder().fit([item for sublist in pos_tags for item in sublist])
    ner_encoder = LabelEncoder().fit([item for sublist in ner_tags for item in sublist])

    encoded_pos = [pos_encoder.transform(pos) for pos in pos_tags]
    encoded_ner = [ner_encoder.transform(ner) for ner in ner_tags]

    X = pad_sequences(sequences, maxlen=max_length, padding="post")
    y_pos = pad_sequences(encoded_pos, maxlen=max_length, padding="post")
    y_ner = pad_sequences(encoded_ner, maxlen=max_length, padding="post")

    return X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder


def save_processed_data(X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder, save_dir):
    """Saves the processed data and encoders to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # Save processed data
    np.save(os.path.join(save_dir, "X.npy"), X)
    np.save(os.path.join(save_dir, "y_pos.npy"), y_pos)
    np.save(os.path.join(save_dir, "y_ner.npy"), y_ner)

    # Save tokenizer and encoders
    with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(save_dir, "pos_encoder.pkl"), "wb") as f:
        pickle.dump(pos_encoder, f)
    with open(os.path.join(save_dir, "ner_encoder.pkl"), "wb") as f:
        pickle.dump(ner_encoder, f)


def load_processed_data(save_dir):
    """Loads the processed data and encoders from disk."""
    X = np.load(os.path.join(save_dir, "X.npy"))
    y_pos = np.load(os.path.join(save_dir, "y_pos.npy"))
    y_ner = np.load(os.path.join(save_dir, "y_ner.npy"))

    with open(os.path.join(save_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(save_dir, "pos_encoder.pkl"), "rb") as f:
        pos_encoder = pickle.load(f)
    with open(os.path.join(save_dir, "ner_encoder.pkl"), "rb") as f:
        ner_encoder = pickle.load(f)

    return X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder


if __name__ == "__main__":
    # Example usage
    filepath = (
        "/home/rafsanbs/code/NERPOS/data/raw/data.tsv"  # Replace with your dataset path
    )
    save_dir = (
        "/home/rafsanbs/code/NERPOS/data/processed"  # Replace with your save directory
    )

    # Load and preprocess data
    data = load_data(filepath)
    X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder = preprocess_data(data)

    # Save processed data and encoders
    save_processed_data(X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder, save_dir)

    # Load processed data (optional)
    (
        X_loaded,
        y_pos_loaded,
        y_ner_loaded,
        tokenizer_loaded,
        pos_encoder_loaded,
        ner_encoder_loaded,
    ) = load_processed_data(save_dir)
