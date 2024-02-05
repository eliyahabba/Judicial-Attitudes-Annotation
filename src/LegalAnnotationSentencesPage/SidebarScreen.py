from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from LegalAnnotationSentencesPage.ExamplesNavigator import ExamplesNavigator


class SidebarScreen:
    def __init__(self, txt_dir_manager):
        self.txt_dir_manager = txt_dir_manager

    @staticmethod
    def display_taggiers_list():
        taggers = [f for f in st.session_state.taggers_data_path.iterdir() if f.is_file()]
        taggers = [f.stem for f in taggers]
        # get the name of the current tagger. Ask the uset to choose his name from the list
        chosen_tagger = st.sidebar.selectbox(
            "Choose your name",
            taggers,
            key="tagger",
        )
        st.session_state["chosen_tagger"] = chosen_tagger

    def create_sidebar(self, ):
        n_files = st.session_state["files_number"]
        n_annotate_files = len(st.session_state["annotation_files"])
        st.sidebar.write("Total sentences:", n_files)
        st.sidebar.write("Total annotate sentences:", n_annotate_files)
        st.sidebar.write("Remaining sentences:", n_files - n_annotate_files)

        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.button(label="Previous sentence for tagging", on_click=ExamplesNavigator.previous_sentence)
        with col2:
            st.button(label="Next sentence for tagging", on_click=ExamplesNavigator.next_sentence)
        # st.sidebar.button(label="Refresh", on_click=self.refresh)

    def refresh(self, show_special_label=None):
        st.session_state["df"] = self.txt_dir_manager.get_df()
        st.session_state["files_number"] = np.arange(len(st.session_state["df"]['title'].tolist()))
        chosen_tagger = st.session_state["chosen_tagger"]

        tagger_df = pd.read_csv(st.session_state.taggers_data_path / f"{chosen_tagger}.csv", dtype={'sentence_id': 'int'})
        if show_special_label is not None:
            tagger_df = tagger_df[
                (tagger_df['label'] == show_special_label) | (tagger_df['label2'] == show_special_label)]
        st.session_state["annotation_files"] = tagger_df[tagger_df["status"] == "annotated"]["title"].tolist()
        st.session_state["file_index"] = 0
