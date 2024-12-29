from bs4 import BeautifulSoup
from typing import List, Tuple, Dict, Optional, Set
from urllib.parse import urlparse, urljoin
import os
import re
import emoji
import html2text
from utils import read_html_from_file

class HTMLProcessor:
    def __init__(self, log_dir: str, output_path: str, filename: str, word_count_threshold: int = 10, url: Optional[str] = None):
        self.log_dir = log_dir
        self.word_count_threshold = word_count_threshold
        self.output_path = output_path
        self.url = url
        self.filename = filename

        # HTML converter setup
        self.html2text = html2text.HTML2Text()
        self.html2text.ignore_links = False
        self.html2text.ignore_images = False
        self.html2text.ignore_tables = False

        # Consolidated selectors
        self.selectors = {
            'content': [
                "article", "main", ".content", "#content", ".post-content",
                ".article-content", ".entry-content", "[role='main']",
                ".main-content", ".post", ".blog-post", ".page-content",
                "#main-content", ".story", ".article-body", ".entry",
                ".post-body", ".content-area", ".site-content",
                "div[itemprop='articleBody']", "[role='article']",
                ".story-content", ".post-article",
                "table", ".table", "#table"
            ],
            'noise': [
                "header", ".header", "#header",
                ".site-header", ".page-header",
                ".advertisement", ".ad-container",
                ".footer", ".site-footer", ".page-footer",
                ".social-share", ".popup", ".modal",
                ".cookie-notice", ".subscription", ".newsletter",
                ".related-articles", ".breadcrumbs", ".search-form",
                ".banner", ".sponsored", ".cookie-banner", ".alert",
                "[role='header']", "[role='footer']"
            ]
        }

    def _log_step(self, step_name: str, content: str, extra_info: str = "") -> None:
        """Centralized logging function"""
        print(f"{step_name}: {extra_info}")
        if self.log_dir:
            self._save_log(step_name, content)

    def _save_log(self, step_name: str, content: str) -> None:
        """Save log to file"""
        try:
            log_path = os.path.join(self.log_dir, f"{step_name}.log")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"Error saving log: {str(e)}")

    def _handle_encoding(self, content: str) -> str:
        """Handle different content encodings"""
        if isinstance(content, str):
            return content

        encodings = ['utf-8', 'iso-8859-1', 'windows-1252', 'ascii']
        for encoding in encodings:
            try:
                if isinstance(content, bytes):
                    return content.decode(encoding)
            except UnicodeDecodeError:
                continue

        return content.decode('utf-8', errors='ignore') if isinstance(content, bytes) else content

    def _get_site_config(self, url: str) -> dict:
        """Get site-specific configuration or default config for unknown sites"""
        try:
            if not url:
                return {'content_selector': None, 'noise_selectors': []}

            domain = urlparse(url).netloc
            if not domain:
                return {'content_selector': None, 'noise_selectors': []}

            # Known site configurations
            configs = {
                'medium.com': {
                    'content_selector': 'article',
                    'noise_selectors': ['.metabar', '.post-tags']
                },
                'wordpress.com': {
                    'content_selector': '.post-content',
                    'noise_selectors': ['.widget-area']
                }
            }

            # Return site config if exists, otherwise return default config
            return configs.get(domain, {'content_selector': None, 'noise_selectors': []})

        except Exception as e:
            print(f"Error in _get_site_config: {str(e)}")
            return {'content_selector': None, 'noise_selectors': []}

    def _extract_links(self, node: BeautifulSoup, base_url: str = "") -> Dict[str, List[str]]:
        """Extract and categorize links from content"""
        internal_links: Set[str] = set()
        external_links: Set[str] = set()

        if base_url:
            base_domain = urlparse(base_url).netloc

            for link in node.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)

                try:
                    parsed_url = urlparse(absolute_url)
                    if parsed_url.netloc == base_domain:
                        internal_links.add(absolute_url)
                    else:
                        external_links.add(absolute_url)
                except Exception:
                    continue

        return {
            "internal": list(internal_links),
            "external": list(external_links)
        }

    def _extract_media(self, node: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract media elements from content"""
        media = {
            "images": [],
            "videos": [],
            "audios": []
        }

        # Extract images
        for img in node.find_all('img', src=True):
            media["images"].append(img['src'])

        # Extract videos
        for video in node.find_all(['video', 'iframe']):
            if video.get('src'):
                media["videos"].append(video['src'])

        # Extract audio
        for audio in node.find_all('audio', src=True):
            media["audios"].append(audio['src'])

        return media

    def _convert_to_markdown(self, node: BeautifulSoup) -> str:
        """Convert HTML content to Markdown"""
        try:
            # Convert to string and pass to html2text
            html_str = str(node)
            self.html2text.body_width = 0  # Disable line wrapping
            return self.html2text.handle(html_str)
        except Exception as e:
            self._log_step("markdown_conversion_error", str(e))
            return ""

    def _detect_main_content(self, body: BeautifulSoup) -> BeautifulSoup:
        """Detect main content area using content density and indicators"""
        for selector in self.selectors['content']:
            content = body.select(selector)
            if content and len(content[0].get_text(strip=True)) > 100:
                return content[0]

        max_score = 0
        main_content = body

        for tag in body.find_all(['div', 'article', 'section', 'table']):
            score = self._calculate_content_score(tag)
            if score > max_score:
                max_score = score
                main_content = tag

        return main_content

    def _calculate_content_score(self, tag: BeautifulSoup) -> float:
        """Calculate content score based on various metrics"""
        text = tag.get_text(strip=True)
        score = len(text)

        if tag.name == 'table':
            rows = tag.find_all('tr')
            if rows:
                data_cells = len(tag.find_all(['td', 'th']))
                score = data_cells * 2 * (1.2 if tag.find('th') else 1)
                if len(text) > 0:
                    score *= (len(text) / data_cells)
        else:
            indicators = ['content', 'article', 'post', 'story', 'main', 'data']
            class_str = ' '.join(tag.get('class', []))
            id_str = tag.get('id', '')

            if any(ind in (class_str + id_str).lower() for ind in indicators):
                score *= 1.2

            words = len(text.split())
            links = len(tag.find_all('a'))
            if links > 0:
                text_link_ratio = words / links
                if text_link_ratio > 2:
                    score *= 1.1

        return score

    def _clean_html(self, node: BeautifulSoup) -> BeautifulSoup:
        """Enhanced HTML cleaning with link preservation"""
        if not node:
            return node

        # Initial link check
        all_links = node.find_all('a')
        self._log_step("initial_links", f"Found {len(all_links)} links")

        def clean_text(text: str) -> str:
            """Clean text content while preserving structure"""
            if not text:
                return ""
            text = emoji.replace_emoji(text, '')
            text = re.sub(r'[:;=]-?[)(/\\|pPoO]', '', text)
            text = re.sub(r'data:image\/[^;]+;base64,[a-zA-Z0-9+/]+=*', '', text)
            return ' '.join(text.split())

        # Remove unwanted elements
        unwanted_tags = {
            'script', 'style', 'meta', 'noscript', 'iframe',
            'next-route-announcer', 'wcm-modal', 'base', 'button',
            'svg', 'title'
        }

        unwanted_attrs = {
            'style', 'onclick', 'onload', 'onmouseover', 'onmouseout',
            'onmouseenter', 'onmouseleave'
        }

        # Clean elements
        for tag in node.find_all(unwanted_tags):
            tag.decompose()

        # Clean attributes and text
        for tag in node.find_all(True):
            if tag:
                # Remove unwanted attributes
                for attr in list(tag.attrs):
                    if attr in unwanted_attrs or attr.startswith(('data-', 'aria-')):
                        del tag.attrs[attr]

                # Clean text
                if tag.string:
                    tag.string = clean_text(tag.string)

        def is_empty(tag):
            """Check if element is empty while preserving links"""
            if tag.name in ['img', 'br', 'hr', 'a']:
                return False
            return not (tag.get_text(strip=True) or
                        tag.find('img') or
                        tag.find('a'))

        # Remove empty elements
        for tag in node.find_all():
            if is_empty(tag):
                tag.decompose()

        # Final link check
        remaining_links = node.find_all('a')
        self._log_step("remaining_links", f"Preserved {len(remaining_links)} links")

        return BeautifulSoup(node.prettify(), 'html.parser')

    def get_content_of_website(self, output_path, filename,url = None, css_selector: str = None) -> dict:
        """Process and extract website content with improved organization"""
        try:
            html_content = read_html_from_file(output_path, filename)
            if not html_content:
                return self._create_empty_result()

                # Initialize processing
            html_content = self._handle_encoding(html_content)
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract and clean content
            body = soup.body or soup
            self._log_step("initial_extraction", "Extracted initial body content")

            # Safely apply configurations - moved noise removal to after configurations
            try:
                site_config = self._get_site_config(url if url else "")
            except Exception as e:
                self._log_step("config_error", f"Error getting site config: {str(e)}")
                site_config = {'content_selector': None, 'noise_selectors': []}

            # Only set css_selector if site_config has one and none was provided
            if site_config.get('content_selector') and css_selector is None:
                css_selector = site_config['content_selector']

            # Remove noise elements
            noise_selectors = self.selectors['noise'].copy()  # Create a copy of default noise selectors
            if site_config.get('noise_selectors'):
                noise_selectors.extend(site_config['noise_selectors'])

            for selector in noise_selectors:
                try:
                    for element in body.select(selector):
                        element.decompose()
                except Exception as e:
                    self._log_step("noise_removal_error", f"Error removing {selector}: {str(e)}")
                    continue

            # Clean HTML
            body = self._clean_html(body)
            self._log_step("clean_html", "Completed HTML cleaning")

            # Detect main content
            try:
                if css_selector:
                    selected_elements = body.select(css_selector)
                    body = selected_elements[0] if selected_elements else self._detect_main_content(body)
                else:
                    body = self._detect_main_content(body)
            except Exception as e:
                self._log_step("content_detection_error", f"Error detecting content: {str(e)}")
                # If content detection fails, use the whole body
                body = body

            # Extract supplementary content
            print("Extract supplementary content")
            result = {
                "markdown": self._convert_to_markdown(body),
                "cleaned_html": str(body),
                "success": True,
                "media": self._extract_media(body),
                "links": self._extract_links(body, url)
            }

            # Save outputs
            if self.output_path:
                self._save_outputs(result)

            return result

        except Exception as e:
            self._log_step("processing_error", f"Error processing content: {str(e)}")
            return self._create_empty_result()

    def _create_empty_result(self) -> dict:
        """Create empty result structure"""
        return {
            "markdown": "",
            "cleaned_html": "",
            "success": False,
            "media": {"images": [], "videos": [], "audios": []},
            "links": {"internal": [], "external": []}
        }

    def _save_outputs(self, result: dict) -> None:
        """Save processed outputs to files"""
        try:
            output_dir = os.path.join(self.output_path, "cleaned_content")
            os.makedirs(output_dir, exist_ok=True)

            print("saving html")
            if result.get("cleaned_html"):
                html_path = os.path.join(output_dir, f"{self.filename}_cleaned.html")
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(result["cleaned_html"])

            print("saving markdown")
            if result.get("markdown"):
                md_path = os.path.join(output_dir, f"{self.filename}.md")
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(result["markdown"])

            self._log_step("save_outputs", f"Saved outputs to {output_dir}")
            print("saved outputs")
        except Exception as e:
            self._log_step("save_outputs_error", str(e))

    def batch_process(self, html_contents: List[Tuple[str, str]]) -> List[Dict]:
        """Batch process multiple HTML contents"""
        return [
            self.get_content_of_website(url, html)
            for url, html in html_contents
            if html
        ]

"""
def main():
        # Create log directory
        log_dir = create_log_directory()

        # Initialize processor with lower threshold for testing
        processor = HTMLProcessor(word_count_threshold=10)

        try:
            # Process content
            result = processor.process_content(
                html_content=html_content,
                url="https://elevenlabs.io",  # Replace with actual URL
                log_dir=log_dir,
                output_path="/content/processed_content/processed_content.md"
            )

            if result and result["content"]["success"]:
                print(f"Successfully processed content")
                print(f"Found {len(result['content']['media']['images'])} images")
                print(f"Found {len(result['content']['links']['internal'])} internal links")
                print(f"Found {len(result['content']['links']['external'])} external links")
            else:
                print("Failed to process content")

        except Exception as e:
            print(f"Error in main: {str(e)}")

    if __name__ == "__main__":
        main()

"""
