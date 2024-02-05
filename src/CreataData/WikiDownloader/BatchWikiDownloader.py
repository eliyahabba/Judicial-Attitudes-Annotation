from pathlib import Path
import pandas as pd
from tqdm import tqdm

from WikiDownloader import WikiDownloader
USER_AGENT = 'YourUserAgent/1.0'
ENGLISH = 'en'
OUTPUT_FOLDER_NAME = 'WikipediaPages'
DATA_FOLDER_NAME = 'WikiData'
DATA_FOLDER_PATH = Path(__file__).parents[3] / DATA_FOLDER_NAME
OUTPUT_FOLDER_PATH = DATA_FOLDER_PATH / OUTPUT_FOLDER_NAME

class BatchWikiDownloader:
    def __init__(self, output_folder=OUTPUT_FOLDER_PATH):
        self.wiki_downloader = WikiDownloader()
        self.output_folder = output_folder

    def download_and_save_pages(self, pages_urls):
        for page_url in tqdm(pages_urls):
            try:
                page = self.wiki_downloader.get_paragraphs(page_url)
                title = page[0][0] # the first heading is the title of the page
                paragraphs = page[1]
                self.save_to_file(title, paragraphs)
                print(f"Successfully saved content for '{title}' to '{title}.txt'")
            except Exception as e:
                print(f"Error while saving content for '{page_url}': {e}")

    def save_to_file(self, title, paragraphs):
        # Ensure the folder 'downloaded_pages' exists (create if not)
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)

        file_path = f"{self.output_folder}/{title}.txt"

        # save the content of the page to a file txt, using the sections to create paragraphs
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(paragraphs))

    def create_paragraphs(self, page):
        paragraphs = []
        for section in page.sections:
            paragraphs.append(section.text)
            paragraphs.extend(self.create_paragraphs(section))
        return paragraphs

# Example Usage:
if __name__ == "__main__":
    # Replace 'YourUserAgent/1.0' with an appropriate user agent
    batch_downloader = BatchWikiDownloader()

    # read the results.csv (the output of the SPIKE retrieval) and extract the titles of the Wikipedia pages
    results_df = pd.read_csv(DATA_FOLDER_PATH / 'results.csv')
    pages_to_download = results_df['article_link'].tolist()

    batch_downloader.download_and_save_pages(pages_to_download)
