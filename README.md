
---

# NLP Pipeline with FastAPI Integration

## Overview

This repository contains an end-to-end machine learning pipeline developed for Named Entity Recognition (NER) and Parts of Speech (POS) tagging tasks. The pipeline includes data preprocessing, model training, evaluation, and deployment using FastAPI. The repository is designed with modularity and best practices in mind, ensuring ease of use and reproducibility.

## Features

- **Data Preprocessing**: Load, clean, tokenize, pad sequences, and split the dataset into training, validation, and test sets.
- **Model Development**: A neural network architecture designed to handle both NER and POS tagging. LSTM BIDIRECTIONAL MODEL
- **Model Evaluation**: Performance metrics including accuracy, precision, recall, and F1 score.
- **API Integration**: FastAPI endpoints to preprocess data, train the model, and evaluate the dataset.

## Project Structure

```bash
.
├── app
│   ├── main.py                # FastAPI app with preprocessing, training, and evaluation endpoints
│   ├── model.py              # Model architecture
│   ├── data_preprocessing.py          # Data preprocessing script
│   ├── train.py               # Model training script
│   └── evaluate.py            # Model evaluation script
│   └── utils.py            # have the common utils code
├── data
│   ├── raw                    # Raw dataset files
│   └── processed              # Preprocessed dataset files
├── models               # Directory for saving trained models
├── requirements.txt           # Python dependencies
├── config.py           # reads the environment file variables
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:BariRafsan/NERPOS.git
   ```


2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add the environment variables according to the .env        template.

5. To run the fastapi script:
   ```bash
   uvicorn main:app --reload 
   ```
 

## Usage

### 1. Data Preprocessing

To preprocess the dataset, send a POST request to the `/preprocess` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/preprocess"
```

This will load, clean, tokenize, pad the sequences. The processed data will be saved in the `data/processed` directory.

### 2. Model Training

To train the model on the preprocessed dataset, use the `/modeltrain` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/modeltrain"
```

The trained model will be saved in the `models` directory.

### 3. Model Evaluation

To evaluate the model and get a performance report, send a POST request to the `/evaluate` endpoint:

```bash
curl -X POST "http://127.0.0.1:8000/evaluate"
```

This will return a JSON object with metrics such as accuracy, precision, recall, and F1 score.

## Model Architecture

The model is a neural network designed to handle both NER and POS tagging tasks. It uses an embedding layer, followed by LSTM layers for sequence processing, and finally, separate dense layers for each task.

## API Endpoints

- **POST /preprocess**: Preprocess the dataset.
- **POST /modeltrain**: Train the model with the preprocessed dataset.
- **POST /evaluate**: Evaluate the model and get performance metrics.

## Future Work

- **Docker Deployment**: Containerize the application for easy deployment.
- **ONNX Integration**: Convert the model to ONNX format for cross-platform inference.

## Contact

For any questions or support, please contact [Rafsan Bari Shafin](mailto:shafinrafsan46@gmail.com).
