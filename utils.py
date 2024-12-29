import os
import json
from datetime import datetime
import html
from bs4 import BeautifulSoup
import re
from urllib.parse import urlparse


class InvalidCSSSelectorError(Exception):
    pass


def extract_filename_from_url(url):
    # Parse the URL
    parsed_url = urlparse(url)

    # Extract the domain and path
    domain = parsed_url.netloc
    path = parsed_url.path

    filename = f"{domain}_{path}".strip("/")
    filename = re.sub(r'[\\/*?"<>|]', '_', filename)

    if filename.endswith("_"):
        filename = filename[:-1]

    max_length = 255
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename
def save_markdown(output_path, filename, markdown_content):
    try:
        # Ensure the directory exists
        filename = filename + ".md"
        os.makedirs(output_path, exist_ok=True)
        full_path = os.path.join(output_path, filename)

        # Write the markdown content
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)

        print(f"Markdown file saved successfully at: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving markdown file: {str(e)}")
        return False


def create_log_directory():
    """Create a timestamped directory for logs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("processing_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def log_state(log_dir, function_name, state_type, state):
    """Log a specific state to a file"""
    filename = os.path.join(log_dir, f"{function_name}.txt")
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"{state_type} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'-' * 50}\n")
        if isinstance(state, dict):
            f.write(json.dumps(state, indent=2, default=str))
        else:
            f.write(str(state))
        f.write("\n")


def beautify_html(escaped_html, log_dir=None):
    # log_state(log_dir, "beautify_html", "INPUT", escaped_html)

    unescaped_html = html.unescape(escaped_html)
    soup = BeautifulSoup(unescaped_html, "html.parser")
    pretty_html = soup.prettify()
    if log_dir:
        log_state(log_dir, "beautify_html", "OUTPUT", pretty_html)
    return pretty_html


def extract_metadata(html_content, log_dir):
    # log_state(log_dir, "extract_metadata", "INPUT", html_content)

    metadata = {}
    if not html_content:
        log_state(log_dir, "extract_metadata", "OUTPUT", metadata)
        return metadata

    soup = BeautifulSoup(html_content, "html.parser")

    # Extract basic metadata
    for tag_name, attr in [
        ("title", None),
        ("description", "name"),
        ("keywords", "name"),
        ("author", "name")
    ]:
        tag = soup.find("title") if tag_name == "title" else \
            soup.find("meta", attrs={"name": tag_name})
        metadata[tag_name] = tag.string if tag_name == "title" and tag else \
            tag["content"] if tag else None

    # Extract Open Graph metadata
    og_tags = soup.find_all("meta", attrs={"property": lambda v: v and v.startswith("og:")})
    for tag in og_tags:
        metadata[tag["property"]] = tag["content"]

    # Extract Twitter Card metadata
    twitter_tags = soup.find_all("meta", attrs={"name": lambda v: v and v.startswith("twitter:")})
    for tag in twitter_tags:
        metadata[tag["name"]] = tag["content"]

    log_state(log_dir, "extract_metadata", "OUTPUT", metadata)
    return metadata


def read_html_from_file(output_path: str,filename) -> str:
    encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'ascii']

    filepath = os.path.join(output_path, "scraped_content", f"{filename}")
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist")
        return ""

    # Try different encodings
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as file:
                content = file.read()
                print(f"Successfully read file using {encoding} encoding")
                print("returning content")
                return content

        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}, trying next encoding")
            continue
        except Exception as e:
            print(f"Error reading file: {str(e)}")
            return ""

    # If all encodings fail, try with error handling
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            print("Read file with utf-8 encoding and error handling")
            return content
    except Exception as e:
        print(f"Final error reading file: {str(e)}")
        return ""


def save_cleaned_html(file_path, filename, content):
    filename = filename + ".html"
    full_path = os.path.join(file_path, filename)
    try:
        with open(full_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Saved cleaned HTML to {full_path}")
    except Exception as e:
        print("error writing cleaned html file:" + str(e))

