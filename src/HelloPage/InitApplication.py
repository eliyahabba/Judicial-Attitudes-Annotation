from pathlib import Path

import PIL.Image as Image
import pandas as pd
import streamlit as st

UTILS_PATH = Path(__file__).parents[1] / "utils"

class InitApplication:

    @staticmethod
    def display_headlines():
        image = Image.open(UTILS_PATH / "logo_image.jpg")
        new_image = image.resize((400, 300))
        title = 'היא היתה נחושה ומשכנעת בתשובותיה והנני נותן אמון מלא בעדותה'
        english_title = 'She was determined and persuasive in her responses, and I hereby place full trust in her testimony'
        judge_name = "השופט אברהם אליקים, 12 יולי 2011"
        judge_name_english = "Judge Avraham Elyakim, July 12, 2011"
        # add an apostrophes to the title in the beginning and in the end
        title = '"' + title + '"'
        english_title = '"' + english_title + '"'

        # write the title in the middle of the page with italic style, and the size of the font is 20
        st.markdown(f'<p style="text-align: center; font-size:20px; font-style:italic;">{title}</p>',
                    unsafe_allow_html=True)
        # write the english title in the middle of the page with italic style, and the size of the font is 20
        st.markdown(f'<p style="text-align: center; font-size:20px; font-style:italic;">{english_title}</p>',
                    unsafe_allow_html=True)
        # write the judge name in the left of the page with italic style, and the size of the font is 14
        st.markdown(f'<p style="text-align: left; font-size:14px; font-style:italic;">{judge_name}</p>',
                    unsafe_allow_html=True)
        # write the english judge name in the left of the page with italic style, and the size of the font is 14
        st.markdown(f'<p style="text-align: left; font-size:14px; font-style:italic;">{judge_name_english}</p>',
                    unsafe_allow_html=True)

        # Display the image in the center of the page
        st.image(new_image)
        # font_utils.create_hebrew_style()

        # Welcome message
        welcome_text = "Welcome and thank you for your assistance."

        # Page options
        page_options_text = "On the left side, you can choose between two pages:"
        categories_text = "1. Annotation Sentences."
        annotations_displayer_text = "2. Annotations Displayer."
        annotation_categories = "3. Examples of Categories."

        # Page descriptions
        page1_description = "On the first page, you will be asked to annotate sentences according to the provided instructions."
        page2_description = "On the second page, you will be able to see the annotations of the other taggers and to compare them to your annotations."
        page3_description = "On the third page, you will be able to see examples of the categories."


        st.markdown(
            f'<h2 style="font-size:30px; color:Blue;">{welcome_text}</h2>',
            unsafe_allow_html=True)

        # Combine all texts
        full_text = f"{page_options_text}\n\n{categories_text}\n\n{annotations_displayer_text}\n\n{annotation_categories}\n\n" \
                    f"{page1_description}\n\n{page2_description}\n\n{page3_description}"


        st.markdown(
            f'<h2 style=" font-size:22px; color:Blue;">{full_text}</h2>',
            unsafe_allow_html=True)

    @staticmethod
    def display_tagger_headlines():
        st.markdown(
            '''<h3 style="text-align: center; font-size:20px; color:Blue;">Please choose your
             name from the taggers list in the left side.</h3>''',
            unsafe_allow_html=True)

    @staticmethod
    def init_session_state(show_special_label=None):
        chosen_tagger = st.session_state["chosen_tagger"]
        tagger_df = pd.read_csv(st.session_state.taggers_data_path / f"{chosen_tagger}.csv", dtype={'sentence_id': 'int'})
        if show_special_label is not None:
            tagger_df = tagger_df[
                (tagger_df['label'] == show_special_label) | (tagger_df['label2'] == show_special_label)]
        if "files_number" not in st.session_state:

            # read df of the chosen tagger
            # get sentences from the chosen tagger
            st.session_state["df"] = tagger_df
            st.session_state["files_number"] = tagger_df.shape[0]
            st.session_state["annotation_files"] = tagger_df[tagger_df["status"] == "annotated"]["title"].tolist()
            st.session_state["file_index"] = 0
            st.session_state["start_sentence_index"] = -1
            st.session_state["end_sentence_index"] = -1
        else:
            st.session_state["df"] = tagger_df
            st.session_state["files_number"] = tagger_df.shape[0]
            st.session_state["annotation_files"] = tagger_df[tagger_df["status"] == "annotated"]["title"].tolist()
