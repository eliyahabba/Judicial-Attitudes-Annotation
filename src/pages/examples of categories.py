# create a streamlit page to display the categories and explanations
import streamlit as st
from utils.Categories import CategoriesManger
from utils.ChooseDataType import ChooseDataType

if __name__ == "__main__":
    ChooseDataType.choose_data_type()
    st.title("Categories")
    st.write("This page displays the categories and exmaples of categories")
    # display the categories and explanations
    categories_manager = CategoriesManger()
    df = categories_manager.df
    df.drop(columns=["category_id", "parent_category_id"], inplace=True)
    # display df and create the index column as the first column (but keep the name of the index column as 'category')
    st.dataframe(df)
