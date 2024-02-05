import pandas as pd
import streamlit as st

class TxtDirManager:
    def __init__(self):
        self._files = []
        self._annotations_files = []

    def save_annotation(self, label, label2, tagger, sentence_id, start, end, mental_state):
        # read the df of the current tagger and save the label to the df

        tagger_df = pd.read_csv(st.session_state.taggers_data_path / f"{tagger}.csv", dtype={'sentence_id': 'int'})
        tagger_df.loc[tagger_df["sentence_id"] == sentence_id, "label"] = label
        tagger_df.loc[tagger_df["sentence_id"] == sentence_id, "label2"] = label2
        tagger_df.loc[tagger_df["sentence_id"] == sentence_id, "status"] = "annotated"
        tagger_df.loc[tagger_df["sentence_id"] == sentence_id, "start_ind"] = start
        tagger_df.loc[tagger_df["sentence_id"] == sentence_id, "end_ind"] = end
        tagger_df.loc[tagger_df["sentence_id"] == sentence_id, "mental_state"] = mental_state

        tagger_df.to_csv(st.session_state.taggers_data_path / f"{tagger}.csv", index=False)
