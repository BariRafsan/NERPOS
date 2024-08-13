import os
from dotenv import load_dotenv

load_dotenv()


dataset_file_path = os.getenv("DATASET_FILE_PATH")
model_dir = os.getenv("MODELS")

processed_data_dir = os.getenv("PROCESSED_DATA_PATH")
