# This is the main function of the app.
# For running the app: "streamlit run app.py"


import streamlit as st


class ExamplesNavigator:
    @staticmethod
    def next_sentence():
        file_index = st.session_state["file_index"]

        if file_index < st.session_state["files_number"] - 1:
            st.session_state["file_index"] += 1
            st.session_state["start_sentence_index"] = -1
            st.session_state["end_sentence_index"] = -1

        else:
            st.warning('This is the last sentence.')

    @staticmethod
    def previous_sentence():
        file_index = st.session_state["file_index"]
        if file_index > 0:
            st.session_state["file_index"] -= 1
            st.session_state["start_sentence_index"] = -1
            st.session_state["end_sentence_index"] = -1
        else:
            st.warning('This is the first sentence.')