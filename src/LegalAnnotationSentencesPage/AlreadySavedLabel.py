import pandas as pd
import streamlit as st

from LegalAnnotationSentencesPage.TextDisplayer import TextDisplayer


class AlreadySavedLabel:
    def __init__(self, text_displayer: TextDisplayer):
        self.text_displayer = text_displayer

    def display_already_saved_label(self):
        # there is a saved label for this sentence
        selected_label = st.session_state["label"]
        selected_label2 = st.session_state["label2"]
        # print a message that the sentence is already annotated

        first_sentence = f"This sentence is already annotated."
        first_sentence_part2 = f" The saved data:"
        second_sentence_part1 = f"The saved labels are:"
        second_sentence_part2 = f"{selected_label}"
        if pd.notna(selected_label2):
            second_sentence_part3 = f"{selected_label2}"
        else:
            second_sentence_part3 = ""

        third_sentence = f"If you want to change the label, select a new label and click on the 'Save' button."

        saved_start_sentence_index = int(st.session_state['df'].iloc[int(st.session_state["file_index"])]['start_ind'])
        saved_end_sentence_index = int(st.session_state['df'].iloc[int(st.session_state["file_index"])]['end_ind'])

        first_sentence_st = f'<p style="font-family:sans-serif;' \
                            f' color:Green; font-size: 16px;">{first_sentence}</p>'
        first_sentence_st_part2 = f'<p style="font-family:sans-serif;' \
                            f' color:Red; font-size: 16px;' \
                                  f'font-weight:bold;><span' \
                                  f'">{first_sentence_part2}</p>'
        second_sentence_st_part1 = f'<p style="text-align:left; color:Red; font-family:sans-serif;  ' \
                             f'font-weight:bold;><span' \
                             f' style=font-weight:bold;>font-size: 16px;">{second_sentence_part1}</p>'
        second_sentence_st_part2 = f'<p style="text-align:center; color:Red; font-family:sans-serif;  ' \
                             f'><span' \
                             f'>font-size: 16px;">{second_sentence_part2}</p>'
        second_sentence_st_part3 = f'<p style="text-align:center; color:Red; font-family:sans-serif;  ' \
                             f'><span' \
                             f'>font-size: 16px;">{second_sentence_part3}</p>'
        third_sentence_st = f'<p style="font-family:sans-serif; color:Green; font-size:' \
                            f' 16px;">{third_sentence}</p>'
        st.markdown(first_sentence_st, unsafe_allow_html=True)
        st.markdown(first_sentence_st_part2, unsafe_allow_html=True)

        self.text_displayer.create_and_display_paragraphs(saved_start_sentence_index,
                                                     saved_end_sentence_index, colour='Red')
        st.markdown(second_sentence_st_part1, unsafe_allow_html=True)
        st.markdown(second_sentence_st_part2, unsafe_allow_html=True)
        st.markdown(second_sentence_st_part3, unsafe_allow_html=True)
        st.markdown(third_sentence_st, unsafe_allow_html=True)
