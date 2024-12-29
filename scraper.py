from firecrawl import FirecrawlApp
import os
from datetime import datetime
from utils import beautify_html, extract_filename_from_url
from dotenv import load_dotenv


class Scraper:
    def __init__(self, output_path):
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from environment variables
        self.output_path = output_path
        self.FIRECRAWL_API = os.getenv('FIRECRAWL_API')
        if not self.FIRECRAWL_API:
            raise ValueError("FIRECRAWL_API key not found in environment variables")

        self.app = FirecrawlApp(api_key=self.FIRECRAWL_API)

    # Rest of the code remains the same
    def scrape(self, url, crawl=False, crawl_limit=0):
        """
        Scrape website content and save to specified directory

        Args:
            url (str): URL to scrape
            output_path (str): Directory path where content should be saved
            crawl (bool): Whether to crawl multiple pages
            crawl_limit (int): Limit of pages to crawl if crawling
        """
        filename = extract_filename_from_url(url)
        # Create output directory if it doesn't exist
        scraped_content_path = os.path.join(self.output_path, "scraped_content")
        os.makedirs(scraped_content_path, exist_ok=True)

        file_path = os.path.join(scraped_content_path, filename)

        if crawl:
            # Crawl multiple pages
            crawl_status = self.app.crawl_url(
                url,
                params={
                    'limit': crawl_limit,
                    'scrapeOptions': {'formats': ['markdown', 'html']}
                },
                poll_interval=30
            )
            print(crawl_status)
            html_content = crawl_status['data'][0]['html']

            # Save crawled content
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(html_content)

            # If there are multiple pages, save them separately
            if len(crawl_status['data']) > 1:
                for i, page in enumerate(crawl_status['data'][1:], 1):
                    page_filename = f"scraped_content_{timestamp}_page_{i}.html"
                    page_path = os.path.join(self.output_path, page_filename)
                    with open(page_path, "w", encoding="utf-8") as file:
                        file.write(page['html'])

        else:
            # Scrape single page
            scrape_result = self.app.scrape_url(url, params={'formats': ['html']})
            html_content = scrape_result['html']

            html_content = beautify_html(html_content)
            # Save scraped content
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(html_content)

        print(f"Content saved to: {self.output_path}")
        return html_content
