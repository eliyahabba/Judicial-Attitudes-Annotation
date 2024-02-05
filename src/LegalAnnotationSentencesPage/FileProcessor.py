import pandas as pd
import streamlit as st

from LegalAnnotationSentencesPage.FileText import FileText


class FileProcessor:
    def __init__(self):
        self.file_text = None
        self.sentence = None

    def load_data(self):
        file_name, sen_id = self.load_example_for_current_index()
        if file_name is None or sen_id is None:
            return

        self.file_text = FileText(file_name, sen_id)
        self.file_text.find_file_path()
        self.file_text.get_paragraphs()

        st.session_state["sentences_to_paragraphs"] = {sentence: i for i, par in enumerate(self.file_text.paragraphs)
                                                       for
                                                       sentence in par}
        st.session_state["sentences"] = self.file_text.sentences

    def load_example_for_current_index(self):
        df = st.session_state["df"]
        if not df.empty:
            row = df.iloc[st.session_state["file_index"]]
        else:
            return None, None
        file_name = row['title']
        annotated_status = row['status']
        if annotated_status == "annotated":
            label = row['label']
            if 'label2' in row:

                label2 = row['label2']
            else:
                label2 = pd.NA
            st.session_state["label"] = label
            st.session_state["label2"] = label2
        else:
            st.session_state["label"] = None
            st.session_state["label2"] = None
        st.session_state['title'] = row['title']
        if 'sentence_id' in st.session_state and row['sentence_id'] != st.session_state['sentence_id']:
            st.session_state["start_sentence_index"] = -1
            st.session_state["end_sentence_index"] = -1
        st.session_state['sentence_id'] = row['sentence_id']
        return file_name, st.session_state['sentence_id']

