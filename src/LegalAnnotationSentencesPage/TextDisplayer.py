import Levenshtein
import streamlit as st

from LegalAnnotationSentencesPage import FileText
from LegalAnnotationSentencesPage.SentenceDisplayer import SentenceDisplayer


class TextDisplayer:
    def __init__(self, file_text: FileText):
        self.file_text = file_text
        self.paragraphs = self.file_text.paragraphs
        self.sentences = self.file_text.sentences
        self.main_sentence = None
        self.n_files = st.session_state["files_number"]

    @staticmethod
    def display_navigate_buttons():
        col1, col2 = st.columns(2)
        with col1:
            st.button(label="Add the previous sentence", on_click=SentenceDisplayer.previous_sentence)
            st.button(label="Remove the first sentence", on_click=SentenceDisplayer.remove_first_sentence)
        with col2:
            st.button(label="Add the next sentence", on_click=SentenceDisplayer.next_sentence)
            st.button(label="Remove the last sentence", on_click=SentenceDisplayer.remove_last_sentence)

    @staticmethod
    def find_relevant_paragraph(all_paragraphs, main_paragraph):
        """
        find the most relevant paragraph to the main paragraph
        :param all_paragraphs:
        :param main_paragraph:
        :return:
        """
        for i, paragraph in enumerate(all_paragraphs):
            if main_paragraph == paragraph:
                return i
        # find the most similar paragraph from the all_paragraphs to the paragraph with text distance
        distances = [Levenshtein.distance(main_paragraph, paragraph) / len(main_paragraph) for paragraph in
                     all_paragraphs]
        min_distance = min(distances)
        return distances.index(min_distance)

    def find_sentence_in_pars(self):
        # find the sentence in the paragraphs
        sentence_index = st.session_state["sentence_index"]
        spike_index = self.file_text.data["global_sentence_index_spike"].index(sentence_index)
        global_sentence_index = self.file_text.data["global_sentence_index_docx"][spike_index]
        st.session_state["sentence_index"] = global_sentence_index
        if st.session_state["start_sentence_index"] < 0 or st.session_state["end_sentence_index"] < 0:
            st.session_state["start_sentence_index"] = global_sentence_index
            st.session_state["end_sentence_index"] = global_sentence_index + 1

    def map_sentences_to_paragraphs(self):
        sentences_tp_pars = {}
        for i, paragraph in enumerate(self.paragraphs):
            sentences = paragraph['sentences']
            for j, sentence in enumerate(sentences):
                sentences_tp_pars[sentence['global_sentence_index']] = sentence
        st.session_state["sentences_to_paragraphs"] = sentences_tp_pars

    @staticmethod
    def aggregated_sentences_to_displaying_paragraphs(start_sentence_index,
                                                      end_sentence_index):
        last_paragraph = -1
        displaying_paragraphs = []
        cur_paragraph = {"prefix": "", "sentence": "", "suffix": ""}
        for i in range(start_sentence_index, end_sentence_index):
            sentence = st.session_state["sentences_to_paragraphs"][i]
            sentence_text = sentence['sentence']
            if sentence['paragraph_index'] != last_paragraph:
                last_paragraph = sentence['paragraph_index']
                displaying_paragraphs.append(cur_paragraph)
                cur_paragraph = {"prefix": "", "sentence": "", "suffix": ""}
            if i == st.session_state["sentence_index"]:
                cur_paragraph["sentence"] = sentence_text
            else:
                if cur_paragraph["sentence"] == "":
                    cur_paragraph["prefix"] += sentence_text + " "
                else:
                    cur_paragraph["suffix"] += sentence_text + " "
        displaying_paragraphs.append(cur_paragraph)
        return displaying_paragraphs

    def create_and_display_paragraphs(self, start_sentence_index,
                                      end_sentence_index, colour="black"):
        displaying_paragraphs = self.aggregated_sentences_to_displaying_paragraphs(
            start_sentence_index=start_sentence_index,
            end_sentence_index=end_sentence_index
        )

        for i, paragraph in enumerate(displaying_paragraphs):
            self.display_paragraph_with_sentence(paragraph, color=colour)

    def display_example_text(self):
        title = 'The text is taken from the file'
        title = f'{title} : {st.session_state["title"]}'
        st.markdown(f'<p style="unicode-bidi: bidi-override;">{title}</p>',
                    unsafe_allow_html=True)
        self.find_sentence_in_pars()
        self.map_sentences_to_paragraphs()
        self.create_and_display_paragraphs(start_sentence_index=st.session_state["start_sentence_index"],
                                           end_sentence_index=st.session_state["end_sentence_index"])

        def go_to_sentence():
            # split the number of the sentence from the string of st.session_state["sentence_for_tagging"]
            # and then convert it to int
            sentence_number = int(st.session_state["sentence_for_tagging"].split(" ")[1]) - 1
            st.session_state["file_index"] = sentence_number
            st.session_state["start_sentence_index"] = -1
            st.session_state["end_sentence_index"] = -1

        st.sidebar.selectbox(
            "Sentences",
            [f"sentence {i}" for i in range(1, self.n_files + 1)],
            index=st.session_state["file_index"],
            on_change=go_to_sentence,
            key="sentence_for_tagging",
        )

    @staticmethod
    def display_paragraph_with_sentence(paragraph, color='grey'):
        text_direction = "LTR" if st.session_state["text_direction"] == "left-to-right" else "RTL"
        st.markdown(f"<p style='text-align:right: "
                    f"input {{unicode-bidi:bidi-override; direction: {text_direction};}}"
                    f" direction: RTL; color: {color}; '>{paragraph['prefix']} <span style=font-weight:bold;>{paragraph['sentence']} </span>{paragraph['suffix']}</p>",
                    unsafe_allow_html=True)
