from LegalAnnotationSentencesPage.InstructionsDisplayerEnglish import InstructionsDisplayerEnglish
from LegalAnnotationSentencesPage.InstructionsDisplayerHebrew import InstructionsDisplayerHebrew


class InstructionsDisplayer:
    def __init__(self, language='hebrew'):
        self.language = language
        self.displayer = InstructionsDisplayerEnglish() if language == "english" else InstructionsDisplayerHebrew()

    def display_instructions(self):
        self.displayer.display_instructions()
