# create df of the taggers and their status for each sentence
import json
import os
from pathlib import Path

import pandas as pd

TAGGERS_NAMES = ["renana", "maya", "carmit", "dana", "itai", "daniel"]
verdict_files_path = Path(__file__).parents[2] / "Data" / "json_docx_v2"
spike_results_path = Path(__file__).parents[2] / "Data" / "spike_results_v8.csv"
taggers_path = Path(__file__).parents[2] / "Data" / "taggers_data_v2"


def split_example_to_tagger(df):
    df_tagger = df.sample(n=100, replace=False)
    df_tagger.drop(columns=["origin_sentence", "paragraph_text"], inplace=True)
    for tagger in TAGGERS_NAMES:
        # sample 100 sentences from df without repetition
        # df_tagger = df.sample(n=100, replace=False)
        df_tagger["tagger"] = tagger
        df_tagger["status"] = "not annotated"

        if os.path.exists(taggers_path / f"{tagger}.csv"):
            assert False, f"file {tagger}.csv already exists"
        else:
            df_tagger.to_csv(taggers_path / f"{tagger}.csv", index=False)

    df2 = df.copy()
    df2.drop(columns=["origin_sentence", "paragraph_text"], inplace=True)
    df2.to_csv(taggers_path / f"full_df.csv", index=False)



def read_current_data():
    dfs = []
    for tagger in TAGGERS_NAMES:
        df = pd.read_csv(taggers_path / f'{tagger}.csv', dtype={'sentence_id': 'int'})
        dfs.append(df)
    return pd.concat(dfs)


def add_new_examples_to_tagger(df):
    current_annotations = read_current_data()
    current_annotations.drop_duplicates(subset=['sentence_id', 'title'], inplace=True)
    create_new_annotations = df[df.origin_sentence.isin(current_annotations.clean_sentence)]

    # select 20 sentences from create_new_annotations without repetition on sentence_id
    create_new_annotations = create_new_annotations.sample(n=40, replace=False)
    create_new_annotations.drop(columns=["origin_sentence", "paragraph_text"], inplace=True)
    create_new_annotations["status"] = "not annotated"
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "../../../sentences_annotation_tool_utils/Data", "taggers_data")

    # allow to select 'good' or 'bad' examples
    # ask the use to decide if the example is good or bad
    y, n = [], []
    for i, row in create_new_annotations.iterrows():
        ans = input(row["clean_sentence"])
        if ans == "y":
            y.append(row)
        else:
            n.append(row)
    dfy = pd.DataFrame(y)

    for tagger in TAGGERS_NAMES:
        df = pd.read_csv(os.path.join(path, f'{tagger}.csv'), dtype={'sentence_id': 'int'})
        create_new_annotations["tagger"] = tagger
        new_df = pd.concat([dfy, df])
        new_df.to_csv(os.path.join(path, f"new_{tagger}.csv"), index=False)


if __name__ == "__main__":
    df = pd.read_csv(
        os.path.join(spike_results_path))

    # create new df for each tagger with 100 random sentences
    split_example_to_tagger(df)

    # dont create new df, just add new sentences to the existing df for each tagger
    # add_new_examples_to_tagger(df)
