import os
import sys
from collections import Counter
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
TAGGERS_NAMES = ["renana", "maya", "carmit", "dana", "itai", "daniel"]

def read_annotated_data():
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    path = os.path.join(dir_path, 'Data', 'taggers_data')
    dfs = []
    for tagger in TAGGERS_NAMES:
        df = pd.read_csv(os.path.join(path, f'{tagger}.csv'), dtype={'sentence_id': 'int'})
        if tagger == 'daniel':
            df['label2'] = pd.NA
        df.drop_duplicates(subset=['sentence_id', 'title', 'tagger'], inplace=True)
        dfs.append(df)
        # if query is nan, it will be int
    return pd.concat(dfs).drop(columns=['query_number'])


def read_not_annotated_data():
    dir_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    path = os.path.join(dir_path, 'Data', 'taggers_data')
    df = pd.read_csv(os.path.join(path, f'full_df.csv'))
    df.drop_duplicates(subset=['sentence_id', 'title'], inplace=True)
    return df


def get_most_label(shared_df):
    shared_df['labels'] = shared_df[['label', 'label' + '2']].apply(lambda x: list(x.dropna()), axis=1)

    # take the most common label for each sentence_id (sentence_id) can appear more than once in the df
    shared_df = shared_df.groupby(['sentence_id']).agg({'labels': lambda x: list(x)})
    # flatten the list of labels
    shared_df['labels'] = shared_df['labels'].apply(lambda x: [item for sublist in x for item in sublist])
    # take the most common label - if the first and second are equal- take the 2 highest labels
    shared_df['labels'] = shared_df['labels'].apply(lambda x: Counter(x).most_common(1) if len(set(x)) == 1
    else Counter(x).most_common(1) if Counter(x).most_common(1)[0][1] > Counter(x).most_common(2)[1][1] else Counter(
        x).most_common(2))

    # split the list of labels - to 2 columns- if there is only one label - the second column will be NaN
    shared_df['label'] = shared_df['labels'].apply(lambda x: x[0])
    shared_df['label2'] = shared_df['labels'].apply(lambda x: x[1] if len(x) == 2 else np.nan)

    # split label to 2 columns - label and label_count
    shared_df['label_count'] = shared_df['label'].apply(lambda x: list(x)[1])
    shared_df['label'] = shared_df['label'].apply(lambda x: list(x)[0])
    # split label to 2 columns - label and label_count
    shared_df['label2_count'] = shared_df['label2'].apply(lambda x: x if pd.isnull(x) else list(x)[1])
    shared_df['label2'] = shared_df['label2'].apply(lambda x: x if pd.isnull(x) else list(x)[0])
    # drop the labels column
    shared_df.drop(columns=['labels'], inplace=True)
    return shared_df

