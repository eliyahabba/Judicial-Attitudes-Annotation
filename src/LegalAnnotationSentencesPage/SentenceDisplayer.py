import streamlit as st


class SentenceDisplayer:
    @staticmethod
    def next_sentence():
        end_sentence_index = st.session_state["end_sentence_index"]
        if end_sentence_index < st.session_state["sentences"] - 1:
            st.session_state["end_sentence_index"] += 1
        else:
            st.warning('This is the last sentence.')

    @staticmethod
    def previous_sentence():
        start_sentence_index = st.session_state["start_sentence_index"]
        if start_sentence_index > 0:
            st.session_state["start_sentence_index"] -= 1
        else:
            st.warning('This is the first sentence.')

    @staticmethod
    def remove_first_sentence():
        start_sentence_index = st.session_state["start_sentence_index"]
        if start_sentence_index < st.session_state["sentence_index"]:
            st.session_state["start_sentence_index"] += 1
        else:
            st.warning('This is the first sentence.')

    @staticmethod
    def remove_last_sentence():
        end_sentence_index = st.session_state["end_sentence_index"]
        if end_sentence_index > st.session_state["sentence_index"] + 1:
            st.session_state["end_sentence_index"] -= 1
        else:
            st.warning('This is the last sentence.')
