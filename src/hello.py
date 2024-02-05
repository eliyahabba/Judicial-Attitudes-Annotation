import streamlit as st

from HelloPage.InitApplication import InitApplication
from utils.ChooseDataType import ChooseDataType


def run():
    st.set_option("deprecation.showfileUploaderEncoding", False)
    InitApplication.display_headlines()


if __name__ == "__main__":
    # Choose the data type (WikiData or LegalData) and set the paths to the data files
    ChooseDataType.choose_data_type()
    run()
