import random
# set random seed
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, \
    SquadExample, squad_convert_examples_to_features
import re

from transformers import AutoModelForQuestionAnswering, BertTokenizer, BertTokenizerFast
import json
import os
from pathlib import Path

import pandas as pd

# from src.AnnotationsProcessing.utils import TAGGERS_NAMES
TAGGERS_NAMES = ["renana", "maya", "carmit", "dana", "itai", "daniel"]


def read_data(data_path):
    dfs = []
    for tagger in TAGGERS_NAMES:
        df = pd.read_csv(data_path / f'{tagger}.csv', dtype={'sentence_id': 'int'})
        dfs.append(df)  # if query is nan, it will be int
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
        dfs.append(df)  # if query is nan, it will be int
    return pd.concat(dfs)


def process_data(df):
    # convert query number to int inplace when it is not nan
    df['query number'] = df['query number'].apply(lambda x: int(x) if not pd.isna(x) else x)
    # remove all row that there label contain 'not relevant' or "רלוונטי"
    df = df[~df['label'].str.contains('not relevant', na=False, case=False)]
    df = df[~df['label'].str.contains('רלוונטי', na=False, case=False)]

    df.drop_duplicates().sort_values('status', key=lambda col: col == 'annotated', ascending=False, inplace=True)
    df.drop_duplicates(subset=['sentence_id', 'title', 'tagger'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates().sort_values('status', key=lambda col: col == 'annotated', ascending=False, inplace=True)
    return df


def add_text_for_the_sentence_v1(unique_sentences_df, query):
    files_path = Path(__file__).parents[2] / 'Data' / 'json_docx'
    tokenizer1 = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    tokenizer2 = BertTokenizer.from_pretrained('amirdnc/HeQ')

    for j, row in unique_sentences_df.iterrows():
        data = get_file_data(row, files_path, version=1)
        global_sentence = data['global_sentence_index']

        context1, answer1 = concat_all_sentences_to_one_text(data, global_sentence,query, tokenizer1)
        context2, answer2 = concat_all_sentences_to_one_text(data, global_sentence,query, tokenizer2)

        # Convert list to one string
        assert answer1 in context1
        assert answer2 in context2


        # add the text to the df with name 'context'
        unique_sentences_df.loc[j, 'context_mbert'] = context1
        unique_sentences_df.loc[j, 'context_heq'] = context2
        unique_sentences_df.loc[j, 'answer_mbert'] = answer1
        unique_sentences_df.loc[j, 'answer_heq'] = answer2
        # add the main_sentence_from_docx to the df with name 'main_sentence_from_docx'
        # unique_sentences_df.loc[j, 'main_sentence_from_docx'] = answer1

    # new_unique_sentences_df = filter_long_examples(unique_sentences_df)
    return unique_sentences_df


def add_text_for_the_sentence_v2(unique_sentences_df, query):
    files_path = Path(__file__).parents[2] / 'Data' / 'json_docx_v2'
    # take the 10 first sentences from the docx file
    tokenizer1 = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
    tokenizer2 = BertTokenizer.from_pretrained('amirdnc/HeQ')

    for j, row in tqdm(unique_sentences_df.iterrows(), total=len(unique_sentences_df)):
        data = get_file_data(row, files_path, version=2)

        global_sentences_index_docx = data['global_sentence_index_docx']
        global_sentences_index_spike = data['global_sentence_index_spike']
        # find the index sentence_id in the global_sentence_index_spike
        if row.sentence_id not in global_sentences_index_spike:
            print(j)
            continue
        global_sentence_index_spike = global_sentences_index_spike.index(row.sentence_id)
        global_sentence = global_sentences_index_docx[global_sentence_index_spike]

        context1, answer1 = concat_all_sentences_to_one_text(data, global_sentence,query, tokenizer1)
        context2, answer2 = concat_all_sentences_to_one_text(data, global_sentence,query, tokenizer2)
        assert answer1 in context1

        assert answer2 in context2
        # add the text to the df with name 'context'
        unique_sentences_df.loc[j, 'context_mbert'] = context1
        unique_sentences_df.loc[j, 'context_heq'] = context2
        # add the main_sentence_from_docx to the df with name 'main_sentence_from_docx'
        unique_sentences_df.loc[j, 'answer_mbert'] = answer1
        unique_sentences_df.loc[j, 'answer_heq'] = answer2
    # new_unique_sentences_df = filter_long_examples(unique_sentences_df)
    return unique_sentences_df

def filter_long_examples(unique_sentences_df, tokenizer):
    examples = []
    for i, row in unique_sentences_df.iterrows():
        context = row.context
        if isinstance(context, float):
            continue
        answer = row.main_sentence_from_docx
        example = SquadExample(
            qas_id=i,
            question_text="כיצד השופט מתייחס אל העדות של המתלוננת?",
            context_text=context,
            answer_text=answer,
            start_position_character=context.index(answer),
            title="title",
        )
        examples.append(example)
    features = squad_convert_examples_to_features(
        examples=examples,
        tokenizer = tokenizer,
        max_seq_length=512,
        doc_stride=128,
        max_query_length=20,
        is_training=True,
        threads=1,
    )
    new_unique_sentences_df = pd.DataFrame()
    for i, f in enumerate(features):
        if i > 0:
            if f.qas_id == features[i - 1].qas_id:
                pass
            else:
                # add the row from the unique_sentences_df to the new_unique_sentences_df
                new_unique_sentences_df = new_unique_sentences_df.append(unique_sentences_df.iloc[f.qas_id ])
        else:
            new_unique_sentences_df = new_unique_sentences_df.append(unique_sentences_df.iloc[f.qas_id])

    return new_unique_sentences_df

def create_qa_df_from_spike_data(query):
    spike_results_path = Path(__file__).parents[2] / "Data" / "spike_results_v8.csv"
    df = pd.read_csv(os.path.join(spike_results_path))
    spike_df = add_text_for_the_sentence_v2(df, query)
    return spike_df

def get_file_data(row: pd.Series, files_path, version: int):
    # get the file name

    file_name = row['title']
    # get the sentence id
    sentence_id = row['sentence_id']
    if version == 1:
        file_prefix = file_name + "_sen_id_" + str(sentence_id) + ".json"
    elif version == 2:
        file_prefix = file_name + ".json"
    else:
        raise ValueError("version must be 1 or 2")
    # read_the file (json)
    file_path = files_path / file_prefix
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def add_point_to_sentece(sentence_text):
    if sentence_text.strip() and len(sentence_text.strip()) >= 2:
        last_char = sentence_text.strip()[-1]
        second_last_char = sentence_text.strip()[-2]
        if last_char not in ["?", "!", ".", ":"] and second_last_char not in ["?", "!", ".", ":"]:
            sentence_text += "."
    elif sentence_text.strip() and len(sentence_text.strip()) == 1:
        last_char = sentence_text.strip()[-1]
        if last_char not in ["?", "!", ".", ":"]:
            sentence_text += "."
    return sentence_text


def concat_all_sentences_to_one_text(data, global_sentence, query, tokenizer):
    all_sentences = []
    for i, paragraph in enumerate(data['paragraphs']):
        sentences = paragraph['sentences']
        text = [sentence['sentence'] for sentence in sentences]
        # recplde all  special spaces to normal space
        text = [re.sub(r'\s+', ' ', sentence) for sentence in text]
        all_sentences.extend(text)

    # take 7 sentence before and 7 after the 'global_sentence'
    global_sentence_paragraph = ". ".join(all_sentences[global_sentence-10:global_sentence+10]) + "."
    # tokenize the global sentence and the tokenize all the paragraph

    text_before = ". ".join(all_sentences[global_sentence-10:global_sentence]) + "."
    text_after = ". ".join(all_sentences[global_sentence+1:global_sentence+10]) + "."
    sentence_text = add_point_to_sentece(all_sentences[global_sentence])
    tokenized_before = tokenizer.encode(text_before, add_special_tokens=False)
    tokenized_after = tokenizer.encode(text_after, add_special_tokens=False)
    tokenized_sentence = tokenizer.encode(sentence_text, add_special_tokens=False)
    tokenized_query = tokenizer.encode(query, add_special_tokens=False)

    # now create a new paragraph with the global sentence in the middle, and the other sentences around it
    # when the size of the paragraph is 490 tokens (the max size of bert is 512) by the tokenized sentences:
    num_of_tokens = 508 - (len(tokenized_sentence))- len(tokenized_query)- 3
    # now we need to find the number of tokens before and after the global sentence
    # take a random number of tokens before (max is num_of_tokens) and the rest after
    num_of_tokens_before = random.randint(0, num_of_tokens)
    num_of_tokens_after = num_of_tokens - num_of_tokens_before

    # now concat the text sentences to one list
    text = text_before + sentence_text + text_after
    global_sentence_paragraph_tokenized = tokenized_before[-num_of_tokens_before:] + tokenized_sentence + tokenized_after[:num_of_tokens_after]
    text_with_exact_tokens = tokenizer.decode(global_sentence_paragraph_tokenized)
    # assert sentence_text in text_with_exact_tokens
    context = text_with_exact_tokens
    answer = tokenizer.decode(tokenized_sentence)
    assert answer in context
    return context, answer


def create_qa_df(unique_df_with_context, query):
    df = pd.DataFrame(columns=['question', 'context', 'answer', 'title', 'sentence_id'])
    for i, row in unique_df_with_context.iterrows():
        # get the file name
        answer_heq = row['answer_heq']
        context_heq = row['context_heq']
        answer_mbert = row['answer_mbert']
        context_mbert = row['context_mbert']
        question = query
        # create the dict
        qa_dict = {'question': question, 'context_mbert': context_mbert,
                      'answer_mbert': answer_mbert, 'context_heq': context_heq,
                        'answer_heq': answer_heq, 'title': row['title'],
                   'sentence_id': row['sentence_id']}
        # add the dict to the df
        df = df.append(qa_dict, ignore_index=True)
    # save the df to csv file
    # df.to_csv('qa_dataset.csv', index=False, encoding='utf-8-sig')
    return df


def combine_Data_from_v1_and_v2(query):
    data_path = Path(__file__).parents[2] / 'Data' / 'taggers_data'
    df = read_data(data_path)
    processed_df = process_data(df)
    processed_df.sort_values('status', key=lambda col: col == 'annotated', ascending=False, inplace=True)
    unique_sentences_df = processed_df.drop_duplicates(subset=['sentence_id', 'title'])
    unique_sentences_df.reset_index(drop=True, inplace=True)

    unique_sentences_df_with_context_v1 = add_text_for_the_sentence_v1(unique_sentences_df, query)

    data_path = Path(__file__).parents[2] / 'Data' / 'taggers_data_v2'
    df = read_data(data_path)
    processed_df = process_data(df)
    processed_df.reset_index(drop=True, inplace=True)
    processed_df.sort_values('status', key=lambda col: col == 'annotated', ascending=False, inplace=True)
    unique_sentences_df = processed_df.drop_duplicates(subset=['sentence_id', 'title'])
    unique_sentences_df.reset_index(drop=True, inplace=True)

    unique_sentences_df_with_context_v2 = add_text_for_the_sentence_v2(unique_sentences_df, query)

    # merge the two df
    unique_sentences_df_with_context = pd.concat(
        [unique_sentences_df_with_context_v1, unique_sentences_df_with_context_v2])
    unique_sentences_df_with_context.reset_index(drop=True, inplace=True)
    unique_sentences_df_with_context.sort_values('status', key=lambda col: col == 'annotated', ascending=False,
                                                 inplace=True)
    return unique_sentences_df_with_context


def create_qa_df_from_annotated_data(query):
    unique_sentences_df_with_context = combine_Data_from_v1_and_v2(query)
    unique_sentences_df_with_context = unique_sentences_df_with_context[
        unique_sentences_df_with_context['status'] == 'annotated']
    return unique_sentences_df_with_context


def create_qa_df_from_not_annotated_data(query):
    unique_sentences_df_with_context = combine_Data_from_v1_and_v2(query)
    unique_sentences_df_with_context = unique_sentences_df_with_context[
        unique_sentences_df_with_context['status'] != 'annotated']
    return unique_sentences_df_with_context


def create_qa_dataset(df, query="כיצד השופט מתאר את העדות של המתלוננת?"):
    qa_df = create_qa_df(df, query=query)
    return qa_df

def create_negative_dataset(query):
    spike_results_path = Path(__file__).parents[2] / "Data" / "spike_results_v8.csv"

    # group by title and collect the sentences to list
    # for each title create a negative example
    negative_df = pd.DataFrame(columns=['question', 'context', 'answer', 'title', 'sentence_id'])
    files_path = Path(__file__).parents[2] / 'Data' / 'json_docx_v2'
    for i,file in enumerate(tqdm(os.listdir(files_path))):
        try:
            file_path = os.path.join(files_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)


            global_sentences_index_docx = data['global_sentence_index_docx']
            # sort the list
            global_sentences_index_docx.sort()
            all_sentences = []
            for i, paragraph in enumerate(data['paragraphs']):
                sentences = paragraph['sentences']
                text = [sentence['sentence'] for sentence in sentences]
                # recplde all  special spaces to normal space
                text = [re.sub(r'\s+', ' ', sentence) for sentence in text]
                all_sentences.extend(text)
            if global_sentences_index_docx:
                for j in range(20,global_sentences_index_docx[0], 21):
                    # concat all the sentences to one text
                    if j+10 > len(all_sentences):
                        continue
                    context, answer = concat_all_sentences_to_one_text(data, j+10, query)
                    new_row = pd.DataFrame(
                        {'question': query, 'context': context, 'answer': None, 'title': file, 'sentence_id': j}, index=[0])
                    negative_df = pd.concat([negative_df, new_row], ignore_index=True)
                for j in range(global_sentences_index_docx[-1]+10, len(all_sentences)-10, 21):
                    context, answer = concat_all_sentences_to_one_text(data, j+10, query)
                    new_row = pd.DataFrame(
                        {'question': query, 'context': context, 'answer': None, 'title': file, 'sentence_id': j}, index=[0])
                    negative_df = pd.concat([negative_df, new_row], ignore_index=True)
            else:
                for j in range(10, len(all_sentences)-10, 21):
                    context, answer = concat_all_sentences_to_one_text(data, j+10, query)
                    new_row = pd.DataFrame(
                        {'question': query, 'context': context, 'answer': None, 'title': file, 'sentence_id': j}, index=[0])
                    negative_df = pd.concat([negative_df, new_row], ignore_index=True)
        except:
            # print(file) with red color
            print("\033[91m {}\033[00m" .format(file))

    return negative_df
if __name__ == "__main__":
    random.seed(42)
    query = "כיצד השופט מתאר את העדות של המתלוננת?"
    # negative_df = create_negative_dataset(query)
    # # save negative_df to csv
    # negative_df.to_csv('negative_df.csv', index=False)

    # full_qa_df = read_from_spike_data()
    spike_df = create_qa_df_from_spike_data(query)
    annotated_df = create_qa_df_from_annotated_data(query)
    not_annotated_df = create_qa_df_from_not_annotated_data(query)
    qa_spike_df = create_qa_dataset(spike_df)
    qa_annotated_df = create_qa_dataset(annotated_df)
    qa_not_annotated_df = create_qa_dataset(not_annotated_df)
    full_qa_df = pd.concat([qa_spike_df, qa_not_annotated_df])
    full_qa_df.reset_index(drop=True, inplace=True)

    train_test_df = qa_annotated_df.copy()
    train_test_df['data_type'] = 'train'
    # add to  qa_annotated_df all the rows from qa_spike_df that are not in qa_annotated_df with data_type = 'test'
    for i, row in qa_not_annotated_df.iterrows():
        if not row['answer'] in qa_annotated_df['answer'].values:
            row['data_type'] = 'test'
            train_test_df = train_test_df.append(row, ignore_index=True)
    train_test_df.reset_index(drop=True, inplace=True)
    for i, row in full_qa_df.iterrows():
        if not row['answer'] in train_test_df['answer'].values:
            row['data_type'] = 'test'
            train_test_df = train_test_df.append(row, ignore_index=True)
    # convert the all columns to string
    train_test_df = train_test_df.astype(str)

    # convert the data to json format (for the huggingface library) so it can be used for training with datasets library
    train_test_df['context'] = train_test_df['context'].apply(lambda x: x.replace('\n', ''))
    train_test_df['question'] = train_test_df['question'].apply(lambda x: x.replace('\n', ''))
    train_test_df['answer'] = train_test_df['answer'].apply(lambda x: x.replace('\n', ''))
    train_test_df['title'] = train_test_df['title'].apply(lambda x: x.replace('\n', ''))
    train_test_df['sentence_id'] = train_test_df['sentence_id'].apply(lambda x: x.replace('\n', ''))
    train_test_df.rename(columns={'answer': 'answers'}, inplace=True)
    train_test_df.rename(columns={'sentence_id': 'id'}, inplace=True)
    train_test_df['data_type'] = train_test_df['data_type'].apply(lambda x: x.replace('\n', ''))
    # split the df to train and test
    train_df = train_test_df[train_test_df['data_type'] == 'train']
    test_df = train_test_df[train_test_df['data_type'] == 'test']
    train_df.drop('data_type', axis=1, inplace=True)
    test_df.drop('data_type', axis=1, inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    # save the df to json file
    train_df.to_csv('spike_data_train.csv', index=False)
    test_df.to_csv('spike_data_test.csv', index=False)
