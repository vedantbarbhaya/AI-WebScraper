import argparse
import asyncio
import sys

from scraper import Scraper
from utils import create_log_directory, extract_filename_from_url
from formatter import HTMLProcessor
from contentProcessor import contentProcessor


def main(url: str):
    # Initialize scraper
    output_path = "data/"
    filename = extract_filename_from_url(url)
    print(f"Filename derived from URL: {filename}")

    # ------------------------- Scraping content -------------------------
    scraper = Scraper(output_path)
    html_content = scraper.scrape(url)

    # Create log directory
    log_dir = create_log_directory()

    # ------------------------- Processing scrapped content -------------------------
    # Initialize processor with lower threshold for testing
    cleaner = HTMLProcessor(
        word_count_threshold=10,
        output_path=output_path,
        filename=filename,
        log_dir=log_dir,
        url=url
    )

    try:
        result = cleaner.get_content_of_website(
            output_path=output_path,
            filename=filename,
            url=url
        )

        if result and result["success"]:
            print("Successfully processed content")
            print(f"Found {len(result['media']['images'])} images")
            print(f"Found {len(result['links']['internal'])} internal links")
            print(f"Found {len(result['links']['external'])} external links")
        else:
            print("Failed to process content")

    except Exception as e:
        print(f"Error in main: {str(e)}")

    # ------------------------- Converting processed content into RAG friendly format -------------------------
    Processor = contentProcessor()
    asyncio.run(Processor.process_html_to_markdown(filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape a single URL.")
    parser.add_argument(
        "--url",
        type=str,
        default="https://en.wikipedia.org/wiki/Srinivasa_Ramanujan",
        help="The URL to scrape. Defaults to Srinivasa Ramanujan's Wikipedia page."
    )
    args = parser.parse_args()

    # Execute main function with provided or default URL
    main(args.url)
