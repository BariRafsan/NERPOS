from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    TimeDistributed,
    Bidirectional,
)


def build_model(vocab_size, max_length, pos_tag_count, ner_tag_count):
    input = Input(shape=(max_length,))
    model = Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length)(
        input
    )
    model = Bidirectional(
        LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
    )(model)

    # POS Tagging
    pos_output = TimeDistributed(
        Dense(pos_tag_count, activation="softmax"), name="pos_output"
    )(model)

    # NER Tagging
    ner_output = TimeDistributed(
        Dense(ner_tag_count, activation="softmax"), name="ner_output"
    )(model)

    model = Model(inputs=[input], outputs=[pos_output, ner_output])
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model
