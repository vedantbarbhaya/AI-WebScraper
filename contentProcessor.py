import tiktoken
import json
from typing import List, Dict, Optional, Tuple, Union
import re
from dotenv import load_dotenv
from prompts import PROMPT_EXTRACT_BLOCKS, SINGLE_CHUNK_INFO, CHUNKED_INFO, SINGLE_OUTPUT_FORMAT, CHUNK_OUTPUT_FORMAT, \
    FIRST_CHUNK_CONTINUATION, MIDDLE_CHUNK_CONTINUATION, FINAL_CHUNK_CONTINUATION
import asyncio
import os
from openai import AsyncOpenAI
from utils import save_markdown


class contentProcessor:
    def __init__(self, model_name: str = "gpt-4o-mini", max_tokens_per_request: int = 180000):
        """
        Initialize the contentProcessor.

        Args:
            model_name: The name of the GPT model being used (for token counting)
            max_tokens_per_request: Maximum tokens allowed per request
        """
        load_dotenv()
        self.max_tokens = 180000  # max tokens per request
        #self.max_tokens = 5000  # max tokens per request
        self.requests_per_minute = 450  # Setting safe limit below 500
        self.last_request_time = 0

        # Initialize tokenizer
        self.encoder = tiktoken.encoding_for_model("gpt-4o-mini")  # or your specific model
        self.themes = None

        # Get API key from environment variables
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Using gpt-4o-mini for higher token limits
        self.output_dir = "my_scraped_content/blocks"
        self.html_content = ""

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoder.encode(text))

    def split_by_sentences(self, text: str) -> List[str]:
        """Split text into chunks by sentences when a segment is too large."""
        sentences = re.split(r'([.!?]+\s+)', text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            if current_tokens + sentence_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks

    def split_by_natural_breaks(self, content: str) -> List[Tuple[str, int]]:
        """Split content at natural HTML break points while respecting token limits."""
        # Split at major HTML structural elements
        parts = re.split(r'(</(?:p|div|section|article|main|header|footer|nav|aside)>)', content)

        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_number = 1

        for part in parts:
            part_tokens = self.count_tokens(part)

            if part_tokens > self.max_tokens:
                # If part is too large, split by sentences
                sentence_chunks = self.split_by_sentences(part)
                for sentence_chunk in sentence_chunks:
                    chunks.append((sentence_chunk, chunk_number))
                    chunk_number += 1
            elif current_tokens + part_tokens > self.max_tokens:
                if current_chunk:
                    chunks.append((''.join(current_chunk), chunk_number))
                    chunk_number += 1
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens

        if current_chunk:
            chunks.append((''.join(current_chunk), chunk_number))

        return chunks

    def create_chunks(self, html_content: str) -> List[Tuple[str, int]]:
        """
        Split HTML content into chunks using natural breaks based on token limit.
        Returns list of (chunk_content, chunk_number) tuples.
        """
        # First check if content needs chunking at all
        total_tokens = self.count_tokens(html_content)
        if total_tokens <= self.max_tokens:
            return [(html_content, 1)]

        # If content needs chunking, use split_by_natural_breaks
        chunks = self.split_by_natural_breaks(html_content)

        # Validate chunks
        validated_chunks = []
        for chunk_content, chunk_number in chunks:
            # If a chunk is still somehow too large (shouldn't happen due to sentence splitting)
            chunk_tokens = self.count_tokens(chunk_content)
            if chunk_tokens > self.max_tokens:
                # Log warning for debugging
                print(f"Warning: Chunk {chunk_number} exceeds token limit. Size: {chunk_tokens}")
                # Split the oversized chunk further if needed
                start_idx = 0
                tokens = self.encoder.encode(chunk_content)
                while start_idx < len(tokens):
                    end_idx = start_idx + self.max_tokens
                    if end_idx > len(tokens):
                        end_idx = len(tokens)

                    sub_chunk = self.encoder.decode(tokens[start_idx:end_idx])
                    validated_chunks.append((sub_chunk, chunk_number))
                    chunk_number += 1
                    start_idx = end_idx
            else:
                validated_chunks.append((chunk_content, chunk_number))

        return validated_chunks

    def extract_themes(self, llm_response: str) -> Dict:
        """
        Extract themes from LLM response.
        Expects themes in JSON format within <themes> tags.
        """
        try:
            # Find themes section in response
            themes_start = llm_response.find('<themes>')
            themes_end = llm_response.find('</themes>')

            if themes_start != -1 and themes_end != -1:
                themes_json = llm_response[themes_start + 8:themes_end].strip()
                self.themes = json.loads(themes_json)
                return self.themes
            else:
                return {}
        except json.JSONDecodeError:
            print("Warning: Could not parse themes JSON")
            return {}
        except Exception as e:
            print(f"Error extracting themes: {str(e)}")
            return {}

    def get_themes_prompt(self) -> str:
        """
        Generate the themes section for the prompt based on previously extracted themes.
        """
        if not self.themes:
            return ""

        return f"""
        Previous themes and sections:
        <previous_themes>
        {json.dumps(self.themes, indent=2)}
        </previous_themes>
        """

    def generate_prompt(self, content_info: dict) -> str:
        """
        Generate appropriate prompt based on content info.

        Args:
            content_info: Dictionary containing content and processing metadata
        """
        if not content_info.get("is_chunked"):
            # Single chunk processing
            return PROMPT_EXTRACT_BLOCKS.format(
                URL="[URL]",
                CHUNK_INFO="",
                HTML=content_info["content"],
                PREVIOUS_THEMES="",
                CHUNK_INSTRUCTIONS="Process the entire content as a single document.",
                OUTPUT_FORMAT=SINGLE_OUTPUT_FORMAT
            )
        else:
            # Chunked processing
            chunk_info = CHUNKED_INFO.format(
                CHUNK_NUMBER=content_info["chunk_number"],
                TOTAL_CHUNKS=content_info["total_chunks"]
            )

            # Determine continuation message
            if content_info["is_last_chunk"]:
                continuation = FINAL_CHUNK_CONTINUATION
            elif content_info["is_first_chunk"]:
                continuation = FIRST_CHUNK_CONTINUATION
            else:
                continuation = MIDDLE_CHUNK_CONTINUATION

            # Set chunk-specific instructions
            if content_info["is_first_chunk"]:
                chunk_instructions = "This is the first chunk. Start the document structure."
            elif content_info["is_last_chunk"]:
                chunk_instructions = "This is the final chunk. Complete any ongoing sections."
            else:
                chunk_instructions = "This is a continuation chunk. Maintain consistency with previous themes and sections."

            output_format = CHUNK_OUTPUT_FORMAT.format(
                CHUNK_NUMBER=content_info["chunk_number"],
                CHUNK_CONTINUATION=continuation
            )

            return PROMPT_EXTRACT_BLOCKS.format(
                URL="[URL]",
                CHUNK_INFO=chunk_info,
                HTML=content_info["content"],
                PREVIOUS_THEMES=content_info.get("themes_prompt", ""),
                CHUNK_INSTRUCTIONS=chunk_instructions,
                OUTPUT_FORMAT=output_format
            )

    def process_content(self, html_content: str) -> Union[Dict, List[Dict]]:
        """
        Process content and prepare it for LLM requests.
        """
        total_tokens = self.count_tokens(html_content)

        if total_tokens <= self.max_tokens:
            return {
                "content": html_content,
                "is_chunked": False,
                "total_tokens": total_tokens
            }

        # Content needs to be chunked
        chunks = self.create_chunks(html_content)
        processed_chunks = []

        for chunk_content, chunk_number in chunks:
            chunk_info = {
                "content": chunk_content,
                "chunk_number": chunk_number,
                "is_first_chunk": chunk_number == 1,
                "is_last_chunk": chunk_number == len(chunks),
                "themes_prompt": self.get_themes_prompt(),
                "is_chunked": True,
                "total_chunks": len(chunks),
                "tokens": self.count_tokens(chunk_content)
            }
            processed_chunks.append(chunk_info)

        return processed_chunks

    async def process_with_openai(self, prompt: str) -> str:
        """
        Process content with OpenAI's API.

        Args:
            prompt: The prompt to send to OpenAI

        Returns:
            The processed response
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that processes HTML content into structured markdown."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Lower temperature for more consistent output
                max_tokens=4000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Extract the response content
            return response.choices[0].message.content

        except Exception as e:
            print(f"Error in OpenAI processing: {str(e)}")
            raise

    async def process_with_llm(self, result: dict):
        try:
            final_output = []
            accumulated_themes = {
                "existing_sections": [],
                "current_themes": [],
                "ongoing_topics": [],
                "terminology": []
            }

            if isinstance(result, dict):
                # Single content processing
                prompt = self.generate_prompt(result)
                print("This is the current prompt:\n" + prompt)
                response = await self.process_with_openai(prompt)
                if '<chunk_content>' not in response:
                    # If response doesn't have chunk tags, wrap it
                    response = f"<chunk_content>\n{response}\n</chunk_content>"
                final_output.append(response)
            else:
                # Chunked content processing
                for i, chunk in enumerate(result):
                    # Add accumulated themes to chunk info
                    if i > 0:  # Not for first chunk
                        chunk['themes_prompt'] = self.get_themes_prompt()

                    # Generate and send prompt
                    prompt = self.generate_prompt(chunk)
                    print("current prompt:\n")
                    print(prompt)
                    response = await self.process_with_openai(prompt)

                    # Extract and accumulate themes
                    chunk_themes = self.extract_themes(response)
                    if chunk_themes:
                        # Update accumulated themes
                        for key in accumulated_themes:
                            if key in chunk_themes:
                                accumulated_themes[key].extend(chunk_themes[key])
                                # Remove duplicates while preserving order
                                accumulated_themes[key] = list(dict.fromkeys(accumulated_themes[key]))

                        # Update chunker's themes for next iteration
                        self.themes = accumulated_themes

                    # Store the processed chunk
                    final_output.append(response)

            # Combine all outputs
            combined_output = self.combine_outputs(final_output)
            return combined_output

        except Exception as e:
            print(f"Error processing content: {str(e)}")
            raise

    def combine_outputs(self, outputs: List[str]) -> str:
        combined_content = []

        continuation_markers = [
            "[Continued in next chunk...]",
            "[Continuing previous section...]",
            "[Continued from previous chunk...]"
        ]

        for output in outputs:
            if not output:  # Skip empty outputs
                continue

            # Initialize chunk_content
            chunk_content = ""

            # Handle output without chunk tags
            if '<chunk_content>' not in output:
                chunk_content = output.strip()
            else:
                # Extract content between <chunk_content> tags
                chunk_start = output.find('<chunk_content>') + len('<chunk_content>')
                chunk_end = output.find('</chunk_content>')

                if chunk_start != -1 and chunk_end != -1:
                    chunk_content = output[chunk_start:chunk_end].strip()

                    # Remove chunk number tag if present
                    chunk_num_start = chunk_content.find('<chunk>')
                    chunk_num_end = chunk_content.find('</chunk>')
                    if chunk_num_start != -1 and chunk_num_end != -1:
                        chunk_content = (
                                chunk_content[:chunk_num_start].strip() +
                                chunk_content[chunk_num_end + len('</chunk>'):].strip()
                        )

            # Only add chunk_content if it's not empty
            if chunk_content:
                # Remove continuation markers
                for marker in continuation_markers:
                    chunk_content = chunk_content.replace(marker, "").strip()

                combined_content.append(chunk_content)

        combined = '\n'.join(combined_content)
        start_tag, end_tag = "<markdown_content>" , "</markdown_content>"

        if start_tag in combined and end_tag in combined:
            start_index = combined.find(start_tag) + len(start_tag)
            end_index = combined.find(end_tag)
            combined = combined[start_index:end_index].strip()

        return combined

    async def process_html_to_markdown(self, filename) -> str:
        """
        Main function to process HTML file to Markdown using OpenAI.
        """
        try:

            # Read HTML file
            filenamehtml = filename + "_cleaned.html"
            html_file_path = "data/cleaned_content/" + str(filenamehtml)
            print(f"Reading HTML file: {html_file_path}")
            with open(html_file_path, 'r', encoding='utf-8') as f:
                self.html_content = f.read()

        except Exception as e:
            print(f"Error in reading html file: {str(e)}")


        try:
            # Process content
            print("Processing content...")
            result = self.process_content(self.html_content)
            combined_output = await self.process_with_llm(result)

            print("\n\n\n" + str('*' * 50) + "\nCOMBINED OUTPUT\n" + str('*' * 50))

            final_op_path = "data/processed_content/"
            save_markdown(final_op_path, filename, combined_output)

        except Exception as e:
            print(f"Error in processing: {str(e)}")

