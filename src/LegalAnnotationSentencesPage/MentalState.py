# This is the main function of the app.
# For running the app: "streamlit run app.py"
import streamlit as st

class MentalState:
    def __init__(self):
        pass


    def is_this_mental_state(self):
        # add  st.radio to the screen - is this mental state?
        is_this_mental_state = st.radio(
            "Does this sentence describe a mental state of the complainant?", ("No", "Yes"), key=f"is_this_mental_state",
            index=0
            # default="No"
        )
        return is_this_mental_state
