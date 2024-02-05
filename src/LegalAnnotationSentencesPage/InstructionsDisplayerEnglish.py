import streamlit as st

import utils.font_utils as fu


class InstructionsDisplayerEnglish:
    @staticmethod
    def display_instructions():
        # display to screen instructions for the user to annotate the text
        instructions = f'Please label the attached text by selecting the appropriate option. Once you have finished tagging the text, ' \
                       f'click on the "Save Tagging" button. '

        instructions2 = """
        If the sentence does not pertain to the impression or credibility of the complainant, you may choose:
        """
        instructions3 = 'not relevant'

        english_instructions = 'Select the appropriate label for the text.' \
                               ' When you finish tagging the text, click the "Save tagging" button.'
        # display the instructions to the user in Hebrew and English languages one after the other
        fu.create_hebrew_style()

        st.markdown(
            f'<p style= font-size:14px;  color:MediumVioletRed; direction: ltr; unicode-bidi:'
            f' bidi-override;">{instructions}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:14px;color:MediumVioletRed ; font-weight:bold; direction: ltr; unicode-bidi:'
            f' bidi-override;">{instructions2}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:14px; color:DarkTurquoise ; font-weight:bold; direction: ltr; unicode-bidi:'
            f' bidi-override;">{instructions3}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:14px; color:MediumVioletRed  ;direction: ltr; unicode-bidi:'
            f' bidi-override;">{english_instructions}</p>',
            unsafe_allow_html=True)


        # context_instructions = \
        #     """
        #     Use the bottoms:
        #     """
        context_instructions2 = \
            """
            Use the bottoms: Add the previous sentence, Remove the first sentence, Add the next sentence, Remove the last sentence
            """

        context_instructions3 = \
            """
            to control the context of the tagged sentence.
            The tagged sentence will appear in
            """
        context_instructions4 = \
        """
        bold
        """
        context_instructions5 = \
        """
        font, while the rest of the sentences will appear in normal font.
         As part of the tagging process, the sentences of the context that you have chosen to display will also be saved.
        """

        context_general_instructions  = "navigate bottom exsplanation"
        with st.expander(context_general_instructions):
            st.markdown(
                f'<p style="font-size:14px;color:MediumBlue   ; font-weight:bold;  direction: ltr; unicode-bidi:'
                f' bidi-override;">{context_instructions2}</p>',
                unsafe_allow_html=True)
            # f" direction: LTR; color: grey; '>{prefix} <span style=font-weight:bold;>{real_answer} </span>{suffix}</p>",
            st.markdown(
                f"<p style='font-size:14px;color:MediumBlue ; direction: ltr; unicode-bidi:"
                f"bidi-override;'>{context_instructions3} <span style=font-weight:bold;>{context_instructions4} </span>{context_instructions5}</p>",
                unsafe_allow_html=True)
