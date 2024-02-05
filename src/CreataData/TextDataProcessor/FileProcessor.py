import json
import os
from pathlib import Path

import Levenshtein
import pandas as pd
import streamlit as st
from tqdm import tqdm

from src.CreataData.TextDataProcessor.JsonManager import JsonManager
from src.CreataData.TextDataProcessor.TextProcessor import TextProcessor
from src.LegalAnnotationSentencesPage.TextDisplayer import TextDisplayer

TXT_FOLDER_NAME = 'WikipediaPages'
DATA_FOLDER_NAME = 'WikiData'
JSON_FOLDER_NAME = 'json_files'
TXT_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME / TXT_FOLDER_NAME
JSON_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME / JSON_FOLDER_NAME


class FileProcessor:
    @staticmethod
    def create_sentences_index_map(sentences):
        sentence_index_map = {}
        total_sentence_index = 0
        for i, paragraph in enumerate(sentences):
            for j, sentence in enumerate(paragraph):
                sentence_key = f"{sentence}_{i}_{j}"
                sentence_index_map[sentence_key] = total_sentence_index
                total_sentence_index += 1
        return sentence_index_map

    @staticmethod
    def read_file_and_process(file_name):
        paragraphs, sentences = TextProcessor.get_paragraphs_and_sentences(file_name)
        sentence_index_map = FileProcessor.create_sentences_index_map(sentences)
        return sentences, sentence_index_map

    @staticmethod
    def write_file_without_main_sentence(file_name):
        sentences, sentence_index_map = FileProcessor.read_file_and_process(file_name)

        json_file_path = JSON_FOLDER_PATH / f"{file_name}.json"

        if not os.path.exists(json_file_path):
            data = {
                "title": file_name,
                "paragraphs": [
                    {
                        "paragraph_index": i,
                        "sentences": [
                            {
                                "sentence": sentence,
                                "sentence_id": str(j),
                                "paragraph_index": i,
                                "global_sentence_index": sentence_index_map[f"{sentence}_{i}_{j}"]
                            }
                            for j, sentence in enumerate(paragraph)
                        ]
                    }
                    for i, paragraph in enumerate(sentences)
                ],
                "main_sentence_from_spike": [],
                "main_sentence_from_docx": [],
                "main_sentence_id_in_docx": [],
                "global_sentence_index_docx": [],
                "global_sentence_index_spike": [],
                "score_of_similarity": []
            }

            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def create_json_from_txt_file(file_name, sentence_id, main_paragraph, main_sentence=None):
        paragraphs, sentences = TextProcessor.get_paragraphs_and_sentences(file_name)

        paragraph_index = TextDisplayer.find_relevant_paragraph(paragraphs, main_paragraph)
        sentence_index_in_paragraph = TextDisplayer.find_relevant_paragraph(sentences[paragraph_index], main_sentence)

        global_sentence_index = sum(
            len(paragraph) for paragraph in sentences[:paragraph_index]) + sentence_index_in_paragraph

        # Save flattened sentences
        flatten_sentences = [sentence for paragraph in sentences for sentence in paragraph]
        similarity_score = FileProcessor.calculate_similarity_score(flatten_sentences[global_sentence_index].strip(),
                                                                    main_sentence.strip())

        sentence_index_map = FileProcessor.create_sentences_index_map(sentences)

        return JsonManager.create_json(file_name, sentences, flatten_sentences, global_sentence_index, paragraph_index,
                                       sentence_index_in_paragraph, similarity_score, main_sentence, sentence_id,
                                       sentence_index_map)

    @staticmethod
    def calculate_similarity_score(sentence1, sentence2):
        return round(Levenshtein.distance(sentence1, sentence2) / len(sentence2), 2)

    @staticmethod
    def process_csv_dataframe(dataframe, threshold=0.2):
        os.makedirs(JSON_FOLDER_PATH, exist_ok=True)
        problematic_files = []
        succeeded_files = []
        error_count = 0
        similarity_scores = []

        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc="Processing CSV"):
            try:
                main_sentence = row['sentence_text']
                file_title = row['title']
                main_paragraph = row['paragraph_text']
                sentence_id = row['sentence_id']

                similarity_score = FileProcessor.create_json_from_txt_file(file_title, sentence_id, main_paragraph,
                                                                           main_sentence)

                if similarity_score < 0:
                    problematic_files.append(file_title)
                else:
                    succeeded_files.append(file_title)
                similarity_scores.append(similarity_score)

            except FileNotFoundError as ex:
                error_count += 1
                FileProcessor.print_error_message(ex, row['title'])
                continue

        # verdicts are all the files in OUTPUT_FOLDER (it is Path object)
        wiki_pages = [wiki_page.name for wiki_page in TXT_FOLDER_PATH.iterdir()]
        all_wiki_pages = [file.split(".")[0] for file in wiki_pages]
        wiki_pages_without_json = [file for file in all_wiki_pages if file not in succeeded_files]
        for file in tqdm(wiki_pages_without_json):
            try:
                FileProcessor.write_file_without_main_sentence(file)
            except FileNotFoundError as ex:
                continue

        FileProcessor.print_summary(len(dataframe), error_count, problematic_files, similarity_scores, threshold)

    @staticmethod
    def print_error_message(exception, file_title):
        print(f"Error: {exception}\nFile Title: {file_title}")

    @staticmethod
    def print_summary(total_files, error_count, problematic_files, similarity_scores, threshold):
        print(f"There are {len(problematic_files)} problematic files out of {total_files}")
        print(f"There are {error_count} errors out of {total_files}")
        print(f"There are {FileProcessor.count_sentences_above_threshold(similarity_scores, threshold)}"
              f" sentences out of {len(similarity_scores)} with similarity_score > {threshold}")

    @staticmethod
    def count_sentences_above_threshold(similarity_scores, threshold):
        return len([similarity_score for similarity_score in similarity_scores if similarity_score > threshold])


if __name__ == '__main__':
    st.session_state['files_number'] = 0
    df_path = Path(__file__).parents[2] / "WikiData" / "results.csv"
    df = pd.read_csv(df_path)
    FileProcessor.process_csv_dataframe(df)
