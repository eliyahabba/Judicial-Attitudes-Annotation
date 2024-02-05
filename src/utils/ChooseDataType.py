from pathlib import Path

import streamlit as st
from utils.Constants import Constants

DataTypesConstants = Constants.DataTypesConstants

class ChooseDataType:
    @staticmethod
    def choose_data_type():
        if "data_type" in st.session_state:  # use the data type that was already chosen as a default
            data_type = st.sidebar.radio("Choose data type", (DataTypesConstants.WIKI_DATA, DataTypesConstants.LEGAL_DATA),
                                         index=1 if st.session_state.data_type == DataTypesConstants.LEGAL_DATA else 0)
        else:  # use the default data type (WikiData)
            data_type = st.sidebar.radio("Choose data type", (DataTypesConstants.WIKI_DATA, DataTypesConstants.LEGAL_DATA))
        if data_type == DataTypesConstants.LEGAL_DATA:
            st.warning("This data is private")
        st.session_state.data_type = data_type
        st.session_state.jsons_path = Path(__file__).parents[2] / data_type / "json_files"
        st.session_state.taggers_data_path = Path(__file__).parents[2] / data_type / "taggers_data"
        st.session_state.categories_path = Path(__file__).parents[2] / data_type / "Categories.csv"
