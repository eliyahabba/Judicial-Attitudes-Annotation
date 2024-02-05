# create df of the taggers and their status for each sentence
from pathlib import Path

import pandas as pd

NUM_SAMPLES = 100

DATA_FOLDER_NAME = 'WikiData'
JSON_FOLDER_NAME = 'json_files'
JSON_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME / JSON_FOLDER_NAME


class TaggersDataProcessor:
    def __init__(self, df, taggers_names, taggers_path, num_samples=NUM_SAMPLES):
        self.df = df
        self.taggers_names = taggers_names
        self.taggers_path = taggers_path
        self.taggers_path.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples

    def split_example_to_tagger(self):
        # remove row that there is not json file for it in the json folder
        for index, row in self.df.iterrows():
            if not (JSON_FOLDER_PATH / f"{row.title}.json").exists():
                self.df.drop(index=index, inplace=True)

        df_tagger = self.df.sample(n=self.num_samples, replace=False)

        for tagger in self.taggers_names:
            self._create_tagger_file(df_tagger, tagger)

        self.df.to_csv(self.taggers_path.parent / "full_taggers_data.csv", index=False)

    def _create_tagger_file(self, df, tagger):
        df_copy = df.copy()
        df_copy["tagger"] = tagger
        df_copy["status"] = "not annotated"
        df_copy["label"] = ""
        df_copy["label2"] = ""
        df_copy["start_ind"] = ""
        df_copy["end_ind"] = ""
        df_copy["mental_state"] = ""

        tagger_file_path = self.taggers_path / f"{tagger}.csv"
        if tagger_file_path.exists():
            raise FileExistsError(f"File {tagger}.csv already exists")

        df_copy.to_csv(tagger_file_path, index=False)

    def read_current_data(self):
        dfs = []
        for tagger in self.taggers_names:
            df = pd.read_csv(self.taggers_path / f'{tagger}.csv', dtype={'sentence_id': 'int'})
            dfs.append(df)
        return pd.concat(dfs)

    def add_new_examples_to_tagger(self, df, num_samples=40):
        current_annotations = self.read_current_data()
        current_annotations.drop_duplicates(subset=['sentence_id', 'title'], inplace=True)

        create_new_annotations = df[df.origin_sentence.isin(current_annotations.clean_sentence)]
        create_new_annotations = create_new_annotations.sample(n=num_samples, replace=False)

        create_new_annotations["status"] = "not annotated"

        for tagger in self.taggers_names:
            tagger_file_path = self.taggers_path / f'{tagger}.csv'
            new_df = pd.concat([create_new_annotations, pd.read_csv(tagger_file_path, dtype={'sentence_id': 'int'})])
            new_df.to_csv(self.taggers_path / f"new_{tagger}.csv", index=False)
