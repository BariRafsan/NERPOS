from source.model import build_model
from icecream import ic
from datetime import datetime
import config

# Get the current date and time
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the model save path with the timestamp
model_save_path = f"nerpos{timestamp}"


def model_training(
    X_train,
    X_test,
    y_pos_train,
    y_pos_test,
    y_ner_train,
    y_ner_test,
    tokenizer,
    pos_encoder,
    ner_encoder,
):
    model = build_model(
        vocab_size=len(tokenizer.word_index) + 1,
        max_length=50,
        pos_tag_count=len(pos_encoder.classes_),
        ner_tag_count=len(ner_encoder.classes_),
    )
    history = model.fit(
        X_train,
        [y_pos_train, y_ner_train],
        validation_data=(X_test, [y_pos_test, y_ner_test]),
        epochs=20,
        batch_size=32,
    )

    model.save(f"{config.model_dir}/{model_save_path}.keras")
    return model_save_path
