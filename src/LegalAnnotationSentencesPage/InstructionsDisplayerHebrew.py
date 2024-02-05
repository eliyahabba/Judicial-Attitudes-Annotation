import streamlit as st

import utils.font_utils as fu


class InstructionsDisplayerHebrew:
    @staticmethod
    def display_instructions():
        # display to screen instructions for the user to annotate the text
        instructions = 'בחרו את התווית המתאימה לטקסט המצורף. כאשר תסיימו לתייג את הטקסט, לחצו על הכפתור "שמור תיוג".'
        instructions2 = """
        אם המשפט כלל אינו רלוונטי (כלומר, הוא אינו מתייחס לרושם או לאמינות של המתלוננת) תוכלו להשתמש בערך:
        """
        instructions3 = 'not relevant'

        english_instructions = 'Select the appropriate label for the text.' \
                               ' When you finish tagging the text, click the "Save tagging" button.'
        # display the instructions to the user in Hebrew and English languages one after the other
        fu.create_hebrew_style()

        st.markdown(
            f'<p style="text-align: right; font-size:16px;  color:MediumVioletRed; direction: rtl; unicode-bidi:'
            f' bidi-override;">{instructions}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="text-align: right; font-size:18px;color:MediumVioletRed ; font-weight:bold; direction: rtl; unicode-bidi:'
            f' bidi-override;">{instructions2}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="text-align: right; font-size:16px; color:DarkTurquoise ; font-weight:bold; direction: ltr; unicode-bidi:'
            f' bidi-override;">{instructions3}</p>',
            unsafe_allow_html=True)
        st.markdown(
            f'<p style="text-align: right; font-size:16px; color:MediumVioletRed  ;direction: ltr; unicode-bidi:'
            f' bidi-override;">{english_instructions}</p>',
            unsafe_allow_html=True)

        context_instructions = \
        """
        תשתמשו בכפתורים:
        Add the previous sentence
        Remove the first sentence
        Add the next sentence
        Remove the last sentence
        
        בשביל לשלוט בהקשר של המשפט המתויג. 
        המשפט המתויג מופיע בטקסט bold, ושאר המשפטים יופיעו בגופן רגיל. 
        כחלק מהשמירה של התיוג, נשמרים גם המשפטים של ההקשר שבחרתם להציג בשביל לתייג בצורה נכונה. 
        לכן לפני שאתם שומרים, שימו לב שהמשפטים המופיעים לפני ואחרי המשפט באמת נדרשים בשביל ההקשר של המשפט
        """

        context_instructions = \
            """
            תשתמשו בכפתורים:
            """
        context_instructions2 = \
            """
            Add the previous sentence, Remove the first sentence, Add the next sentence, Remove the last sentence
            """

        context_instructions3 = \
            """
            בשביל לשלוט בהקשר של המשפט המתויג.
            המשפט המתויג מופיע בטקסט
            """
        context_instructions4 = \
        """
        dlob
        """
        context_instructions5 = \
        """
            , ושאר המשפטים יופיעו בגופן רגיל.
            כחלק מהשמירה של התיוג, נשמרים גם המשפטים של ההקשר שבחרתם להציג בשביל לתייג בצורה נכונה.
            לכן לפני שאתם שומרים, שימו לב שהמשפטים המופיעים לפני ואחרי המשפט באמת נדרשים בשביל ההקשר של המשפט
        """

        context_general_instructions  = "הסבר על הכפתורים"
        with st.expander(context_general_instructions):
            st.markdown(
                f'<p style="text-align: right; font-size:18px;color:MediumBlue  ; direction: rtl; unicode-bidi:'
                f' bidi-override;">{context_instructions}</p>',
                unsafe_allow_html=True)

            st.markdown(
                f'<p style="text-align: right; font-size:18px;color:MediumBlue   ; font-weight:bold;  direction: ltr; unicode-bidi:'
                f' bidi-override;">{context_instructions2}</p>',
                unsafe_allow_html=True)
            # f" direction: RTL; color: grey; '>{prefix} <span style=font-weight:bold;>{real_answer} </span>{suffix}</p>",
            st.markdown(
                f"<p style='text-align: right; font-size:18px;color:MediumBlue ; direction: rtl; unicode-bidi:"
                f"bidi-override;'>{context_instructions3} <span style=font-weight:bold;>{context_instructions4} </span>{context_instructions5}</p>",
                unsafe_allow_html=True)
