import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.AnnotationsProcessing.utils import TAGGERS_NAMES


def read_data():
    data_path = Path(__file__).parents[2] / 'Data'/  'taggers_data'
    dfs = []
    for tagger in TAGGERS_NAMES:
        df = pd.read_csv(data_path/ f'{tagger}.csv', dtype={'sentence_id': 'int'})
        dfs.append(df)                                                                     # if query is nan, it will be int
    merged_df = pd.concat(dfs)
    return merged_df


def remove_duplicates():
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    path = os.path.join(dir_path, 'Data', 'taggers_data')
    dfs = []
    for tagger in TAGGERS_NAMES:
        df = pd.read_csv(os.path.join(path, f'{tagger}.csv'), dtype={'sentence_id': 'int'})
        df.drop_duplicates(subset=['sentence_id'], inplace=True)
        df.to_csv(os.path.join(path, f'{tagger}.csv'), index=False)
        dfs.append(df)                                                                     # if query is nan, it will be int
    return pd.concat(dfs)


def merge_all_labels(df):
    # create a ned df, where if row in df has 2 labels, we create 2 rows in the new df
    # with the same sentence_id, title, tagger, start_ind, end_ind, status, sentence
    # but with different labels
    df2 = pd.DataFrame(columns=df.columns)
    for i, row in tqdm(df.iterrows()):
        df2 = df2.append(row)
        if pd.notnull(row['label2']):
            row['label'] = row['label2']
            df2 = df2.append(row)
    df2.drop(columns=['label2'], inplace=True)
    return df2

def process_data(df):
    df = df[df['status'] == 'annotated']
    # convert start_ind, end_ind, to int inplace
    df['start_ind'] = df['start_ind'].astype('int')
    df['end_ind'] = df['end_ind'].astype('int')

    # convert query number to int inplace when it is not nan
    df['query number'] = df['query number'].apply(lambda x: int(x) if not pd.isna(x) else x)

    df.drop_duplicates(subset=['sentence_id', 'title', 'tagger'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # remove all row that there label contain 'not relevant' or "רלוונטי"
    df = df[~df['label'].str.contains('not relevant', na=False, case=False)]
    df = df[~df['label'].str.contains('רלוונטי', na=False, case=False)]

    return df


def print_stats(merged_df):
    labels_count = merged_df.groupby('label').count()['sentence_id']

    # pretty print the labels count
    print('labels count:')
    for label, count in labels_count.items():
        print(f'{label}: {count}')


def save_df_for_training(merged_df):
    df2 = merged_df[['sentence_id', 'clean_sentence', 'label']]
    df2.rename(columns={'clean_sentence': 'sentence'}, inplace=True)
    df2.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'Data', 'train_data.csv'),
               index=False)


if __name__ == "__main__":
    df = read_data()
    processed_df = process_data(df)
    unique_sentences_df = processed_df.drop_duplicates(subset=['sentence_id', 'title'])
    # merged_df = merge_all_labels(processed_df)
    # print_stats(merged_df)
    # # now we have a df with 2 rows for each sentence that has 2 labels
    # # we want to count the number of sentences for each label
    save_df_for_training(unique_sentences_df)