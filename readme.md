This project is aimed to scrape any website data and convert it to a format suitable for RAG pipeline.

To use this project,

1. Setup your firecrawl api - https://www.firecrawl.dev and OpenAI API key in .env file.
2. Run the runner.py file like this -
   python main.py --url "https://en.wikipedia.org/wiki/SomeOtherPage"
3. You can integrate the functionality of crawling a website using Firecrawl API. An example code is present in runner.py file.
4. The processed content will be present under /data folder in markdown format suitable for LLM processing.
5. 

 