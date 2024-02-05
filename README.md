# Project Annotation Interface
This repository contains the code for an annotation interface designed for
assessing judicial attitudes toward victims of sexual violence in 
the Israeli court system. The study focuses on analyzing court
statements to evaluate how rape myths resonate in the criminal
justice system's response to sex crimes, particularly in the judicial assessment
of victim credibility.

The interface is part of the research presented in the paper
["The Perfect Victim: Computational Analysis of Judicial Attitudes towards Victims of Sexual Violence"](https://arxiv.org/abs/2305.05302)  
**ICAIL 2023**

For an in-depth understanding of the motivation and details behind this annotation
interface, please refer to the **Data** section in [our paper](https://arxiv.org/abs/2305.05302).

# Table of Contents
* [Background](#Background)
* [Annotation Interface](#Annotation-Interface)
* [Usage and Wikipedia Data](#Usage-and-Wikipedia-Data)
  * [Usage](#Usage)
  * [Testing with Wikipedia Data](#Testing-with-Wikipedia-Data)
* [Run on Your Data](#Run-on-Your-Data)
* [How to run?](#How-to-run?)
* [License](#License)

# Background
The annotation task involves the development of computational models to analyze
court statements and assess judicial attitudes toward victims of sexual violence
. An ontology with eight ordinal labels and binary categorizations is formulated
to evaluate these attitudes based on insights from legal
scholarship on rape myths and judicial assessment.
The goal is to provide a comprehensive understanding of
the various attitudes and biases influencing how judges perceive
and assess victims in sexual assault cases.

# Annotation Interface
The annotation interface addresses legal complexities,
offering a controlled and informed annotation process.
Key features include:

* Display of sentences selected by SPIKE.
* Classification into granular-level categories.
* Control buttons to expand the context window,
allowing for classification with additional context when necessary
(as instructed to annotators).
* Tracking of additional sentences viewed by annotators.
* Annotators can mark sentences as not relevant, aiding in correcting extraction errors and improving data accuracy.

# Usage and Wikipedia Data
## Usage
To test the interface with Wikipedia data, use this link.
The interface is structured to contain two types of data:
legal data and Wikipedia data. Legal data is private and sensitive,
so to access it, you need to contact us.
The categories presented in the article appear inside the legalData folder.
For testing purposes, you can use the provided Wikipedia data.

## Testing with Wikipedia Data
For testing purposes, the interface can be run using Wikipedia data.
The steps below guide you through the process:
1. Use 
[SPIKE](https://spike.apps.allenai.org/datasets/wikipediaBasic/search#query=1;JTdCJTIybWFpbiUyMiUzQSUyMiU3QiU1QyUyMnR5cGUlNUMlMjIlM0ElNUMlMjJTJTVDJTIyJTJDJTVDJTIyY29udGVudCU1QyUyMiUzQSU1QyUyMiUzQVNvbWVvbmUlMjB3YXMlMjAlMjRlZHVjYXRlZCUyMGF0JTIwJTNBc29tZXdoZXJlLiU1QyUyMiU3RCUyMiUyQyUyMmZpbHRlcnMlMjIlM0ElMjIlNUIlNUQlMjIlMkMlMjJjYXNlU3RyYXRlZ3klMjIlM0ElMjJpZ25vcmUlMjIlN0Q=&autoRun=true)
to search for results with a simple sentence structure like  
_**:Someone was $educated at :somewhere.**_
2. Save the results in the [result.csv](https://github.cs.huji.ac.il/gabis-lab/sentences_annotation_tool/blob/master/WikiData/results.csv). 
**Note**: In this data, only the first 500 results were saved for demonstration purposes.
3. Download the corresponding Wikipedia pages using the code provided in
[BatchWikiDownloader.py](https://github.cs.huji.ac.il/gabis-lab/sentences_annotation_tool/blob/master/src/CreataData/WikiDownloader/BatchWikiDownloader.py) .
4. Convert text files to JSON format using [convert_data_json.py](https://github.cs.huji.ac.il/gabis-lab/sentences_annotation_tool/blob/master/src/CreataData/convert_data_json.py)
5. Create data for taggers, providing each with a list of random sentences from the text with
[create_taggers_data.py](https://github.cs.huji.ac.il/gabis-lab/sentences_annotation_tool/blob/master/src/CreataData/create_taggers_data.py).
6. Follow the instructions in [How to run?](#How-to-run?) section to execute the interface.

# Run on Your Data
If you want to use the interface for your own data, follow these steps:

1. Generate data using SPIKE's interface, as explained in the [Testing with Wikipedia Data](#Testing-with-Wikipedia-Data) section, or using another method of your choice.
2. Create a folder of text files with your data.
3. Choose the sentences you want to tag from the documents.
4. Convert the text files to JSON format.
5. Divide the sentences for different taggers. 
6. Add the name of the new data (folder name) to [Constants.py](https://github.cs.huji.ac.il/gabis-lab/sentences_annotation_tool/blob/master/src/utils/Constants.py).
7. Run the Streamlit interface as described in the "How to run?" section.

**Note:** You can refer to the [Testing with Wikipedia Data](#Testing-with-Wikipedia-Data) section for additional guidance on generating data using SPIKE's interface.

# How to run?
To run the streamlit app, run the following command:

> virtualenv venv  
source venv/bin/activate.csh  
pip install -r requirements.txt  
cd src  
python main.py

# License
This project is licensed under the [MIT License](https://github.cs.huji.ac.il/gabis-lab/sentences_annotation_tool/blob/master/LICENSE).

**Note:** The artifacts of this study, including the ontology,
annotated dataset, and computational model, will be made available
upon request to contribute to further research and understanding of
judicial attitudes toward victims of sexual violence in the
Israeli court system. The aim is to foster progress in
this important area and promote a more equitable and just response
to sexual assault within the legal system.