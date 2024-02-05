import requests
from bs4 import BeautifulSoup

URL_EXAMPLE = "https://en.wikipedia.org/wiki/Margaret Mercer Elphinstone"


class WikiDownloader:
    @staticmethod
    def get_paragraphs(url=URL_EXAMPLE):
        # Fetch URL Content
        r = requests.get(url)

        # Get body content
        soup = BeautifulSoup(r.text, 'html.parser').select('body')[0]

        # Initialize variable
        paragraphs = []
        heading = []

        # Iterate through all tags
        remaining_content = []
        for tag in soup.find_all():
            # Check each tag name
            # For Paragraph use p tag
            if tag.name == "p":
                # use text for fetch the content inside p tag
                paragraphs.append(tag.text)
            elif "h" in tag.name:
                if "h1" == tag.name:
                    heading.append(tag.text)
            else:
                remaining_content.append(tag.text)
        return heading, paragraphs


# Example Usage:
if __name__ == "__main__":
    wiki_downloader = WikiDownloader()

    # Replace 'Python (programming language)' with the title of the page you want to download
    page_url = "https://en.wikipedia.org/wiki/Python_(programming_language)"

    heading, paragraphs = wiki_downloader.get_paragraphs(page_url)
    print(heading)