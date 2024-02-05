"""
This script is used to convert the data from txt files to json files.
"""
from pathlib import Path

import pandas as pd

from TextDataProcessor.FileProcessor import FileProcessor

DATA_FOLDER_NAME = 'WikiData'
DATA_FOLDER_PATH = Path(__file__).parents[2] / DATA_FOLDER_NAME

if __name__ == '__main__':
    df_path = DATA_FOLDER_PATH / "results.csv"
    df = pd.read_csv(df_path)
    FileProcessor.process_csv_dataframe(df)
