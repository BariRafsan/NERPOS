from source.model import build_model
from .utils import (
    X_train,
    X_test,
    y_pos_train,
    y_pos_test,
    y_ner_train,
    y_ner_test,
    tokenizer,
    pos_encoder,
    ner_encoder,
)
from icecream import ic
from datetime import datetime

# Get the current date and time
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

# Define the model save path with the timestamp
model_save_path = f"my_model_{timestamp}.keras"


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
    epochs=10,
    batch_size=32,
)

model.save(f"model/{model_save_path}.keras")
