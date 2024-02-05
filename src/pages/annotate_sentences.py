import streamlit as st
from HelloPage.InitApplication import InitApplication
from LegalAnnotationSentencesPage.AddOptionToNewSentence import AddOptionToNewSentence
from LegalAnnotationSentencesPage.AlreadySavedLabel import AlreadySavedLabel
from LegalAnnotationSentencesPage.CategorySaver import CategorySaver
from LegalAnnotationSentencesPage.FileProcessor import FileProcessor
from LegalAnnotationSentencesPage.InstructionsDisplayer import InstructionsDisplayer
from LegalAnnotationSentencesPage.SidebarScreen import SidebarScreen
from LegalAnnotationSentencesPage.TextDisplayer import TextDisplayer
from LegalAnnotationSentencesPage.TxtDirManager import TxtDirManager
from utils.ChooseDataType import ChooseDataType


def run():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    ChooseDataType.choose_data_type()

    st.session_state.text_direction = "left-to-right" if st.session_state.data_type == "WikiData" else "right-to-left"
    txt_dir_manager = TxtDirManager()
    sidebar_screen = SidebarScreen(txt_dir_manager)
    sidebar_screen.display_taggiers_list()
    InitApplication.init_session_state()
    sidebar_screen.create_sidebar()

    instructions_displayer = InstructionsDisplayer("english")
    instructions_displayer.display_instructions()
    file_processor = FileProcessor()
    file_processor.load_data()
    text_displayer = TextDisplayer(file_processor.file_text)
    text_displayer.display_example_text()

    TextDisplayer.display_navigate_buttons()
    if st.session_state["label"]:
        already_saved_label = AlreadySavedLabel(text_displayer)
        already_saved_label.display_already_saved_label()

    category_saver = CategorySaver(txt_dir_manager)
    category_saver.save()

    add_option_to_new_sentence = AddOptionToNewSentence(file_processor)
    add_option_to_new_sentence.add_option_to_new_sentence()


if __name__ == "__main__":
    run()
