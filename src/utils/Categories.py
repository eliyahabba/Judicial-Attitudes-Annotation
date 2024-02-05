# read Categories.csv file and create a class for each category
import os

# read csv file with pandas
import pandas as pd
import streamlit as st


# create a Categories class
class CategoriesManger:
    def __init__(self):
        self.categories_to_explanations = None
        self.path = st.session_state.categories_path
        self.df = pd.read_csv(self.path)
        self.create_pairs_categories_and_explanations()

    def create_pairs_categories_and_explanations(self):
        categories = self.df['category'].tolist()
        explanations = self.df['sentence_example'].tolist()

        self.categories_to_explanations = dict(zip(categories, explanations))
