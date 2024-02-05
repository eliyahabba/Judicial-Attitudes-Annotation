import streamlit as st

import utils.font_utils as fu
from LegalAnnotationSentencesPage.ExamplesNavigator import ExamplesNavigator
from LegalAnnotationSentencesPage.MentalState import MentalState
from utils import font_utils
from utils.Categories import CategoriesManger


class CategorySaver:
    def __init__(self, txt_dir_manager):
        self.categories_manger = CategoriesManger()
        self.labels = list(self.categories_manger.categories_to_explanations.keys())
        self.txt_dir_manager = txt_dir_manager

    def save2(self):
        form = st.form("checkboxes", clear_on_submit=True)
        with form:
            msg = "choose the best category for the sentence."
            # print the msg to the user
            st.markdown(
                f"<p style='text-align:LTR; input {{unicode-bidi:bidi-override; direction: RTL;}}"
                f" direction: LTR; color: grey; 'font-weight:bold;><span style=font-weight:bold;>"
                f" {msg} </span></p>",
                unsafe_allow_html=True)

            url_for_examples_category = "https://docs.google.com/spreadsheets/d/13obiQ8eLHANjnwJorlesYVYTKQDAZo-v6h1R8M9TZ6E/edit#gid=0"
            st.write(
                f"You can look at the [examples file]({url_for_examples_category}) to see examples for each category.")
            default_index = len(self.labels) - 1

            selected_label = st.selectbox(
                "Main category", self.labels, key=f"selectedLabel",
                index=default_index
            )
            st.session_state["selected_label"] = selected_label
            second_selected_instruction = "Secondary category - Choose the secondary category for" \
                                          " the sentence (if the sentence is fits to more than one category)."
            with st.expander(second_selected_instruction):
                selected_label2 = st.selectbox(
                    "Secondary category", self.labels + [None], key=f"selectedLabel2",
                    index=len(self.labels)
                )
                st.session_state["selected_label2"] = selected_label2

            # print to the screen the selected label
            font_utils.create_en_style()
            output1 = f"The selected labels are:"
            # output2 = f"Main category:"
            # output3 = f" {selected_label}"
            output4 = f"Secondary category:"
            output5 = f" {selected_label2}"
            fu.markdown_not_bold_text_to_html(output1)
            # fu.markdown_not_bold_and_bold_text_to_html(output2, output3)
            if selected_label2:
                fu.markdown_not_bold_and_bold_text_to_html(output4, output5)
            # allow the user to write a comment
            # resize the comment box
            # output_text = "Comment - you can write a comment here. For example," \
            #               " if you think that this is not a good sentence, you can write it here."
            # fu.create_hebrew_style()
            # st.text_area(output_text, height=200, max_chars=1000)
            mental_state = MentalState()
            is_this_mental_state = mental_state.is_this_mental_state()
            st.session_state["mental_state"] = is_this_mental_state
            submit = form.form_submit_button(label="Save", on_click=self.annotate,
                                             args=(selected_label, selected_label2, is_this_mental_state))

    def save(self):
        with st.container():
            msg = "choose the best category for the sentence."
            # print the msg to the user
            st.markdown(
                f"<p style='text-align:LTR; input {{unicode-bidi:bidi-override; direction: RTL;}}"
                f" direction: LTR; color: grey; 'font-weight:bold;><span style=font-weight:bold;>"
                f" {msg} </span></p>",
                unsafe_allow_html=True)

            default_index = len(self.labels) - 1

            selected_label = st.selectbox(
                "Main category", self.labels, key=f"selectedLabel",
                index=default_index
            )
            st.session_state["selected_label"] = selected_label
            second_selected_instruction = "Secondary category - Choose the secondary category for" \
                                          " the sentence (if the sentence is fits to more than one category)."
            with st.expander(second_selected_instruction):
                selected_label2 = st.selectbox(
                    "Secondary category", self.labels + [None], key=f"selectedLabel2",
                    index=len(self.labels)
                )
                st.session_state["selected_label2"] = selected_label2

            # print to the screen the selected label
            font_utils.create_en_style()
            output1 = f"The selected labels are:"
            output2 = f"Main category:"
            output3 = f" {selected_label}"
            output4 = f"Secondary category:"
            output5 = f" {selected_label2}"
            # with col1:
            fu.markdown_not_bold_text_to_html(output1)
            fu.markdown_not_bold_and_bold_text_to_html(output2, output3)
            if selected_label2:
                fu.markdown_not_bold_and_bold_text_to_html(output4, output5)
                # allow the user to write a comment

            mental_state = MentalState()
            is_this_mental_state = mental_state.is_this_mental_state()
            st.session_state["mental_state"] = is_this_mental_state
            st.button(label="Save", on_click=self.annotate)

    def not_annotate(self):
        st.warning("You must choose a label")

    def annotate(self):
        if "selected_label" not in st.session_state or st.session_state["selected_label"] is None:
            st.warning("You must choose a label")

        self.txt_dir_manager.save_annotation(st.session_state["selected_label"],
                                             st.session_state["selected_label2"],
                                             st.session_state["tagger"],
                                             st.session_state["sentence_id"],
                                             st.session_state["start_sentence_index"],
                                             st.session_state["end_sentence_index"],
                                             st.session_state["mental_state"])
        self.next_annotate_file()

    @staticmethod
    def next_annotate_file():
        file_index = st.session_state["file_index"]
        if file_index < st.session_state["files_number"] - 1:
            st.session_state["file_index"] += 1
            st.session_state["start_sentence_index"] = -1
            st.session_state["end_sentence_index"] = -1
        else:
            st.warning("All sentences are annotated.")
            ExamplesNavigator.next_sentence()
