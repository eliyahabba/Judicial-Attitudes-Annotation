import json
import random

import Levenshtein
import pandas as pd
import streamlit as st


class AddOptionToNewSentence:
    def __init__(self, file_processor):
        self.full_df = None
        self.full_df_path = None
        self.file_processor = file_processor
        self.get_full_df()

    def create_new_example(self, selected_sentence_without_dot):
        sentence_id = random.randint(100000000, 999999999)  # create random sentence_id
        sentence_params = {
            "sentence_id": sentence_id,
            "title": st.session_state["title"],
            "query_number": None,
            "clean_sentence": selected_sentence_without_dot,
            "tagger": st.session_state["tagger"],
            "status": "not annotated",
            "label": None,
            "label2": None,
            "start_ind": None,
            "end_ind": None
        }
        return sentence_params

    def add_option_to_new_sentence(self):
        # ask user if there is another sentence to annotate from the same file
        with st.expander("Do you want to add another sentence to the same file?"):
            # ask the user if the another sentence is instead of the current one, or in addition to it,
            sentences = []
            for i in range(st.session_state["start_sentence_index"], st.session_state["end_sentence_index"]):
                sentence = st.session_state["sentences_to_paragraphs"][i]
                sentences.append(sentence['sentence'])

            # display the sentences in selectbox
            selected_sentence = st.selectbox("Select the sentence to annotate", ['<select>'] + sentences)
            if selected_sentence != '<select>':
                # get the sentence index
                sentence_index = sentences.index(selected_sentence) + st.session_state["start_sentence_index"]
                # display the chosen sentence
                st.markdown(f"The chosen sentence is:")
                st.markdown(
                    f'<p style="text-align: right; font-size:18px;color:MediumBlue  ; direction: rtl; unicode-bidi:'
                    f' bidi-override;">{selected_sentence}</p>',
                    unsafe_allow_html=True)

                # create a dict with the params:
                # sentence_id,title,query number,clean_sentence,tagger,status,label,start_ind,end_ind
                # So we can add it to the df directly
                selected_sentence_without_dot = selected_sentence[:selected_sentence.rfind('.')]
                sentence_params = self.create_new_example(selected_sentence_without_dot)

                sentence_is_already_in_df = self.new_sentence_is_similar_to_existing(
                    sentence_params["clean_sentence"])
                if sentence_is_already_in_df:
                    st.write("The sentence is already in the full df")
                else:
                    self.save_new_sentence(sentence_params, sentence_index)

    def save_new_sentence(self, sentence_params, sentence_index):
        # add the sentence to the df in the after row number of the current sentence, st.session_state["file_index"],
        # without removing the row of the index
        # move all the rows after the current sentence to the next row
        self.full_df.drop(columns=['similarity'], inplace=True)
        df1 = st.session_state["df"].iloc[:st.session_state["file_index"] + 1]
        df2 = st.session_state["df"].iloc[st.session_state["file_index"] + 1:]
        df1_with_new_sentence = df1.append(sentence_params, ignore_index=True)
        st.session_state["df"] = pd.concat([df1_with_new_sentence, df2]).reset_index(drop=True)

        # add the sentence to the df in the current sentence row
        st.session_state["df"].loc[st.session_state["file_index"] + 1, :] = sentence_params
        # # ask the use if he sure with his answer with a botton
        if st.button("SAVE - Are you sure?"):
            chosen_tagger = st.session_state["chosen_tagger"]
            taggers_path = st.session_state.taggers_data_path / (chosen_tagger + '.csv')

            # we need to create a new json file to the new sentence
            # we take the current json file, copy it, and add the new sentence to the copy instead of the current one
            # we need to create a new json file to the new sentence
            cur_file_name = st.session_state["file_name"]
            cur_file_path = st.session_state.JSONS_PATH / cur_file_name

            # new_file_name = cur_file_name + "_sen_id_" + str(sentence_params['sentence_id']) + ".json"
            # new_file_path = os.path.join(VERDICT_PATH, new_file_name)
            data = self.file_processor.file_text.data

            id = str(st.session_state["sentences_to_paragraphs"][sentence_index]['paragraph_index']) + '_' + \
                 str(st.session_state["sentences_to_paragraphs"][sentence_index]['sentence_id'])
            data["main_sentence_from_spike"].append(None)
            data["main_sentence_from_docx"].append(st.session_state["sentences_to_paragraphs"][sentence_index][
                                                       'sentence'])
            data["main_sentence_id_in_docx"].append(id)
            data["global_sentence_index_docx"].append(sentence_index)
            data["global_sentence_index_spike"].append(sentence_params['sentence_id'])
            # -1 if min(data["global_sentence_index_spike"]) >= 0 else min(data["global_sentence_index_spike"]) - 1)
            data["score_of_similarity"].append(0.0)
            # write the new data to the file

            with open(cur_file_path, 'w', encoding="utf-8") as outfile:
                json.dump(data, outfile, ensure_ascii=False, indent=4)
            # change the permissions of the new file to read and write, so the annotator can edit it
            # os.chmod(cur_file_path, 0o775)

            # add the new sentence to the tagger file
            # save the new sentence to the taggers csv file
            st.session_state["df"].to_csv(taggers_path, index=False)

            # add the row to the full df (with keys from the full_df columns).
            # take to sentence_params only the keys that are in the full_df columns
            full_df_columns = self.full_df.columns
            # change name of key 'clean_sentence' to 'sentence_text
            sentence_params['sentence_text'] = sentence_params.pop('clean_sentence')
            sentence_params = {key: value for key, value in sentence_params.items() if
                               key in full_df_columns}
            full_df = self.full_df.append(sentence_params, ignore_index=True)
            full_df.to_csv(self.full_df_path, index=False)

    def new_sentence_is_similar_to_existing(self, selected_sentence_without_dot):
        # check if the sentence is already in the full df (by title, clean_sentence)
        # if it is, then display a message to the user that the sentence is already in the full df
        # if it is not, then add the sentence to the full df
        self.full_df['similarity'] = self.full_df['sentence_text'].apply(
            lambda x: Levenshtein.distance(x, selected_sentence_without_dot) / len(
                selected_sentence_without_dot))
        already_in_df = self.full_df[(self.full_df['title'] == st.session_state["title"]) & (
                self.full_df['similarity'] < 0.1)]

        return not already_in_df.empty

    def get_full_df(self):
        self.full_df_path = st.session_state.taggers_data_path.parent / 'full_taggers_data.csv'
        self.full_df = pd.read_csv(self.full_df_path, dtype={'sentence_id': 'int'})
