import streamlit as st
import utils.font_utils as fu


class InstructionsDisplayerEnglish:
    @staticmethod
    def display_instructions():
        # Display instructions for the user to annotate the text
        main_instructions = (
            "Please label the attached text by selecting the appropriate option. "
            "Once you have finished tagging the text, click on the 'Save Tagging' button."
        )

        additional_instructions = (
            "If the sentence does not pertain to the impression or credibility of the complainant, "
            "you may choose:"
        )
        option_not_relevant = "not relevant"

        english_instructions = (
            "Select the appropriate label for the text. "
            "When you finish tagging the text, click the 'Save tagging' button."
        )

        # Display the instructions to the user in Hebrew and English languages one after the other
        fu.create_hebrew_style()

        st.markdown(
            f'<p style="font-size:14px; color:MediumVioletRed; direction: ltr; unicode-bidi:'
            f' bidi-override;">{main_instructions}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size:14px; color:MediumVioletRed; font-weight:bold; direction: ltr; '
            f'unicode-bidi: bidi-override;">{additional_instructions}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-size:14px; color:DarkTurquoise; font-weight:bold; direction: ltr; unicode-bidi:'
            f' bidi-override;">{option_not_relevant}</p>',
            unsafe_allow_html=True,
        )


        # Context instructions
        context_instructions = (
            "Use the buttons: 'Add the previous sentence', 'Remove the first sentence', "
            "'Add the next sentence', 'Remove the last sentence' to control the context of the tagged sentence. "
            "The tagged sentence will appear in "
        )
        bold_text = "bold"
        context_instructions_continued = (
            " font, while the rest of the sentences will appear in normal font. "
            "As part of the tagging process, the sentences of the context that you have chosen to display "
            "will also be saved."
        )

        context_general_instructions = "Navigate bottom explanation"
        with st.expander(context_general_instructions):
            st.markdown(
                f'<p style="font-size:14px; color:MediumBlue; font-weight:bold; direction: ltr; '
                f'unicode-bidi: bidi-override;">{context_instructions}</p>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:14px; color:MediumBlue; direction: ltr; unicode-bidi:"
                f"bidi-override;'>{bold_text}</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<p style='font-size:14px; color:MediumBlue; direction: ltr; unicode-bidi:"
                f"bidi-override;'>{context_instructions_continued}</p>",
                unsafe_allow_html=True,
            )
