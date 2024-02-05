import json
import os
from pathlib import Path

TXT_FOLDER_NAME = 'WikipediaPages'
DATA_FOLDER_NAME = 'WikiData'
JSON_FOLDER_NAME = 'json_files'
TXT_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME / TXT_FOLDER_NAME
JSON_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME / JSON_FOLDER_NAME


class JsonManager:
    @staticmethod
    def create_json_path(file_name):
        return JSON_FOLDER_PATH / f"{file_name}.json"

    @staticmethod
    def create_json_data(file_name, sentences, sentence_index_map, flatten_sentences, global_sentence_index, par_ind,
                         sen_ind_in_par, similarity_score, main_sentence, sentence_id):
        paragraphs_data = [
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
        ]

        return {
            "title": file_name,
            "paragraphs": paragraphs_data,
            "main_sentence_from_spike": [main_sentence],
            "main_sentence_from_docx": [flatten_sentences[global_sentence_index]],
            "main_sentence_id_in_docx": [f"{par_ind}_{sen_ind_in_par}"],
            "global_sentence_index_docx": [global_sentence_index],
            "global_sentence_index_spike": [sentence_id],
            "score_of_similarity": [similarity_score]
        }

    @staticmethod
    def write_json_to_file(json_file_path, data):
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def create_json(file_name, sentences, flatten_sentences, global_sentence_index, par_ind, sen_ind_in_par,
                    similarity_score,
                    main_sentence, sentence_id, sentence_index_map):
        json_file_path = JsonManager.create_json_path(file_name)

        if not os.path.exists(json_file_path):
            data = JsonManager.create_json_data(file_name, sentences, sentence_index_map, flatten_sentences,
                                                global_sentence_index,
                                                par_ind, sen_ind_in_par, similarity_score, main_sentence, sentence_id)
            JsonManager.write_json_to_file(json_file_path, data)
        else:
            JsonManager.update_existing_json_file(json_file_path, flatten_sentences, global_sentence_index, par_ind, sen_ind_in_par,
                                                  sentence_id, similarity_score,
                                                  main_sentence)

        return similarity_score

    @staticmethod
    def update_existing_json_file(json_file_path, flatten_sentences, global_sentence_index, par_ind, sen_ind_in_par,
                                  sentence_id, similarity_score, main_sentence):
        data = json.load(open(json_file_path, "r", encoding="utf-8"))
        if global_sentence_index in data["global_sentence_index_docx"] or sentence_id in data[
            "global_sentence_index_spike"]:
            return similarity_score

        data["main_sentence_from_spike"].append(main_sentence)
        data["main_sentence_from_docx"].append(flatten_sentences[global_sentence_index])
        data["main_sentence_id_in_docx"].append([f"{par_ind}_{sen_ind_in_par}"])
        data["global_sentence_index_docx"].append(global_sentence_index)
        data["global_sentence_index_spike"].append(sentence_id)
        data["score_of_similarity"].append(similarity_score)

        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
