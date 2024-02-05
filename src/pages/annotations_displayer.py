from AnnotationsDisplayerPage.AnnotatedResultsDisplayer import AnnotatedResultsDisplayer
from utils.ChooseDataType import ChooseDataType



def run():
    ChooseDataType.choose_data_type()
    annotated_results_displayer = AnnotatedResultsDisplayer()
    annotated_results_displayer.display_results()

if __name__ == "__main__":
    run()
