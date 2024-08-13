from contextlib import asynccontextmanager

from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI
import config
from source.data_preprocessing import (
    load_data,
    preprocess_data,
    save_processed_data,
    load_processed_data,
)
from source.utils import train_test_split_func
from source.train import model_training
from source.evaluate import model_evaluate
app = FastAPI()


origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/preprocess")
async def data_preprocessing_and_save():

    X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder = preprocess_data(
        load_data(config.dataset_file_path)
    )
    saved_process_data = save_processed_data(
        X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder, config.processed_data_dir
    )

    return {
        "message": f"Data Preprocessing and saving is done successfully in here {saved_process_data}"
    }


@app.get("/modeltrain")
async def model_train():
    print(config.processed_data_dir)
    X, y_pos, y_ner, tokenizer, pos_encoder, ner_encoder = await load_processed_data(
        config.processed_data_dir
    )
    X_train, X_test, y_pos_train, y_pos_test, y_ner_train, y_ner_test = (
        await train_test_split_func(X, y_pos, y_ner)
    )
    result = model_training(
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
    if result:

        return {"message": "Model Training is done successfully"}
    return {"message": "failed"}


app.get("/evaluate")
async def model_evaluate():
    