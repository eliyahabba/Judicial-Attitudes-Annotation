import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)


class AnnotatedResultsDisplayer:
    def __init__(self):
        self.taggers_data_path = st.session_state.taggers_data_path
        self.df_of_all_taggers = None
        self.read_data()

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a UI on top of a dataframe to let viewers filter columns

        Args:
            df (pd.DataFrame): Original dataframe

        Returns:
            pd.DataFrame: Filtered dataframe
        """
        modify = st.checkbox("Add filters", value=True)

        if not modify:
            return df

        df = df.copy()

        # Try to convert datetimes into a standard format (datetime, no timezone)
        for col in df.columns:
            if is_object_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                except Exception:
                    pass

            if is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)

        modification_container = st.container()

        with modification_container:
            cols = df.columns
            to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
            for column in to_filter_columns:
                left, right = st.columns((1, 20))
                # Treat columns with < 10 unique values as categorical
                if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].unique(),
                        default=list(df[column].unique()),
                    )
                    df = df[df[column].isin(user_cat_input)]
                elif is_numeric_dtype(df[column]):
                    _min = float(df[column].min())
                    _max = float(df[column].max())
                    step = (_max - _min) / 100
                    user_num_input = right.slider(
                        f"Values for {column}",
                        min_value=_min,
                        max_value=_max,
                        value=(_min, _max),
                        step=step,
                    )
                    df = df[df[column].between(*user_num_input)]
                elif is_datetime64_any_dtype(df[column]):
                    user_date_input = right.date_input(
                        f"Values for {column}",
                        value=(
                            df[column].min(),
                            df[column].max(),
                        ),
                    )
                    if len(user_date_input) == 2:
                        user_date_input = tuple(map(pd.to_datetime, user_date_input))
                        start_date, end_date = user_date_input
                        df = df.loc[df[column].between(start_date, end_date)]
                else:
                    user_text_input = right.text_input(
                        f"Substring or regex in {column}",
                    )
                    if user_text_input:
                        df = df[df[column].astype(str).str.contains(user_text_input)]

        return df

    @st.cache_data
    def convert_df(_self, df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8-sig')


    def display_results(self):

        st.title("Auto Filter Annotated Results")
        filtered_df = self.filter_dataframe(self.df_of_all_taggers)
        st.dataframe(filtered_df, width=1000,
                     height=500)
        # add a download button to download the filtered dataframe
        csv = self.convert_df(filtered_df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='df.csv',
            mime='text/csv',
        )

    def read_data(self):

        dfs = []
        taggers = [f for f in self.taggers_data_path.iterdir() if f.is_file()]
        taggers = [f.stem for f in taggers]
        for tagger in taggers:
            df = pd.read_csv(self.taggers_data_path / f'{tagger}.csv', dtype={'sentence_id': 'int'})
            dfs.append(df)
        # Concatenate all dataframes
        self.df_of_all_taggers = pd.concat(dfs, ignore_index=True)
        # reset index
        # self.df_of_all_taggers.reset_index(drop=True, inplace=True)
        self.df_of_all_taggers.rename(columns={'clean_sentence': 'sentence'}, inplace=True)
        # self.df_of_all_taggers.drop(columns=['query_number'], inplace=True)
        # change column order
        self.df_of_all_taggers = self.df_of_all_taggers[['sentence_text', 'label', 'label2', 'tagger',
                                                         'start_ind', 'end_ind', 'status', 'mental_state']]
        # convert to int if not nan
        self.df_of_all_taggers['start_ind'] = self.df_of_all_taggers['start_ind'].apply(lambda x:
                                                                                        str(int(x)) if not pd.isna(
                                                                                            x) else x)
        self.df_of_all_taggers['end_ind'] = self.df_of_all_taggers['end_ind'].apply(lambda x:
                                                                                    str(int(x)) if not pd.isna(
                                                                                        x) else x)
