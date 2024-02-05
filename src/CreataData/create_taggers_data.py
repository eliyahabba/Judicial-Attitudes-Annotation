import argparse
from pathlib import Path

import pandas as pd

from TaggersData.TaggersDataProcessor import TaggersDataProcessor

NUM_SAMPLES = 100
TAGGERS_NAMES = ["Tagger1", "Tagger2", "Tagger3", "Tagger4", "Tagger5", "Tagger6"]
DATA_FOLDER_NAME = 'WikiData'
RESULTS_PATH = Path(__file__).parents[2] / DATA_FOLDER_NAME / "results.csv"
TAGGERS_PATH = Path(__file__).parents[2] / DATA_FOLDER_NAME / "taggers_data"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--taggers", nargs="+", default=TAGGERS_NAMES, help="list of taggers")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES, help="number of samples to take from the df")
    args = parser.parse_args()
    df = pd.read_csv(RESULTS_PATH)

    # create new df for each tagger with 100 random sentences
    taggers_data_processor = TaggersDataProcessor(df, args.taggers, TAGGERS_PATH, num_samples=args.num_samples)
    taggers_data_processor.split_example_to_tagger()

    # don't create new df, just add new sentences to the existing df for each tagger
    # add_new_examples_to_tagger(df)
