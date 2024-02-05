import re
from pathlib import Path

TXT_FOLDER_NAME = 'WikipediaPages'
DATA_FOLDER_NAME = 'WikiData'
TXT_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME / TXT_FOLDER_NAME


class TextProcessor:
    @staticmethod
    def split_paragraph_to_sentence(paragraph):
        split_chars = ["!", "?", "."]

        # Use a regular expression to match any of the split characters
        split_regex = r"(?<!\d)(" + "|".join(re.escape(char) for char in split_chars) + ")"

        sentences = re.split(split_regex, paragraph)

        # Filter out empty sentences and those without alphabetic characters
        sentences = [sentence.strip() for sentence in sentences if
                     sentence and any(char.isalpha() for char in sentence)]

        return sentences

    @staticmethod
    def read_text_file(file_name):
        file_path = TXT_FOLDER_PATH / f"{file_name}.txt"
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    @staticmethod
    def process_paragraphs(file_text):
        paragraphs = [paragraph.strip() for paragraph in file_text.split('\n') if paragraph]
        return paragraphs

    @staticmethod
    def process_paragraphs_to_sentences(paragraphs):
        sentences = [TextProcessor.split_paragraph_to_sentence(paragraph) for paragraph in paragraphs]
        # clean the citation marks from the sentences, this is done by removing the text between '[' and ']'
        cleaned_sentences = [[re.sub(r'\[.*?\]', '', sentence) for sentence in paragraph] for paragraph in sentences]
        return cleaned_sentences

    @staticmethod
    def get_paragraphs_and_sentences(file_name):
        file_text = TextProcessor.read_text_file(file_name)
        paragraphs = TextProcessor.process_paragraphs(file_text)
        sentences = TextProcessor.process_paragraphs_to_sentences(paragraphs)
        return paragraphs, sentences
