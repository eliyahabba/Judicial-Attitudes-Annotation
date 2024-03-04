import streamlit as st
import utils.font_utils as fu

class InstructionsDisplayerHebrew:
    @staticmethod
    def display_instructions():
        # Display instructions for the user to annotate the text
        main_instructions = (
            "בחרו את התווית המתאימה לטקסט המצורף. כאשר תסיימו לתייג את הטקסט, לחצו על הכפתור 'שמור תיוג'."
        )

        additional_instructions = (
            "אם המשפט כלל אינו רלוונטי (כלומר, הוא אינו מתייחס לרושם או לאמינות של המתלוננת) תוכלו להשתמש "
            "בערך:"
        )
        option_not_relevant = "not relevant"

        fu.create_hebrew_style()

        st.markdown(
            f'<p style="text-align: right; font-size:16px; color:MediumVioletRed; direction: rtl; unicode-bidi: bidi-override;">'
            f"{main_instructions}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="text-align: right; font-size:18px; color:MediumVioletRed; font-weight:bold; direction: rtl; '
            f'unicode-bidi: bidi-override;">{additional_instructions}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="text-align: right; font-size:16px; color:DarkTurquoise; font-weight:bold; direction: ltr; unicode-bidi:'
            f' bidi-override;">{option_not_relevant}</p>',
            unsafe_allow_html=True,
        )

        context_instructions_general = \
            """
            תשתמשו בכפתורים:
            """
        context_instructions_bottoms = \
            """
            Add the previous sentence, Remove the first sentence, Add the next sentence, Remove the last sentence
            """

        context_instructions_control = \
            """
            בשביל לשלוט בהקשר של המשפט המתויג.
            המשפט המתויג מופיע בטקסט
            """
        context_instructions_bold = \
        """
        dlob
        """
        context_instructions_explain = \
        """
            , ושאר המשפטים יופיעו בגופן רגיל.
            כחלק מהשמירה של התיוג, נשמרים גם המשפטים של ההקשר שבחרתם להציג בשביל לתייג בצורה נכונה.
            לכן לפני שאתם שומרים, שימו לב שהמשפטים המופיעים לפני ואחרי המשפט באמת נדרשים בשביל ההקשר של המשפט
        """

        context_general_instructions  = "הסבר על הכפתורים"
        with st.expander(context_general_instructions):
            st.markdown(
                f'<p style="text-align: right; font-size:18px;color:MediumBlue  ; direction: rtl; unicode-bidi:'
                f' bidi-override;">{context_instructions_general}</p>',
                unsafe_allow_html=True)

            st.markdown(
                f'<p style="text-align: right; font-size:18px;color:MediumBlue   ; font-weight:bold;  direction: ltr; unicode-bidi:'
                f' bidi-override;">{context_instructions_bottoms}</p>',
                unsafe_allow_html=True)
            # f" direction: RTL; color: grey; '>{prefix} <span style=font-weight:bold;>{real_answer} </span>{suffix}</p>",
            st.markdown(
                f"<p style='text-align: right; font-size:18px;color:MediumBlue ; direction: rtl; unicode-bidi:"
                f"bidi-override;'>{context_instructions_control} <span style=font-weight:bold;>{context_instructions_bold} </span>{context_instructions_explain}</p>",
                unsafe_allow_html=True)
