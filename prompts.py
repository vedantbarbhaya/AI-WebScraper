PROMPT_EXTRACT_BLOCKS  = """Here is the URL of the webpage:
<url>{URL}</url>

And here is the {CHUNK_INFO}content:
<html>
{HTML}
</html>

{PREVIOUS_THEMES}

Your task is to transform this content into a well-structured Markdown document optimized for knowledge retrieval. The purpose of this activity is 
to transform a website content into a format suitable for retrieval. Keeping that in mind, decide on how much summarization is optimal in order to 
remove unwanted content but keep content suited for retrieval and question answering tasks downstream.


{CHUNK_INSTRUCTIONS}

General Guidelines:
1. Create a document with:
   - # [Main Title] - Start with the main topic/product name
   - Brief overview of what this is about. Do not limit yourself to certain number of sentence. Do not oversimplify or over-summarize the content.
   - Use ## for major sections based on content's natural organization
   - Use bullet points (-) for listing related items
   - Bold (**text**) for key terms and important concepts
   - Maintain factual accuracy and original terminology
   - Remove links but preserve the names/text of linked entities
   - Remove any images/image_links but keep associated company/entity names
   - Try to keep proper nouns or names of entities intact

2. When describing:
   - Features/Capabilities: Focus on what it does and key benefits
   - Technical aspects: Be precise but accessible
   - Products/Solutions: Highlight main use cases and target users
   - Partnerships/Integrations: Note key relationships and their significance

3. Format Guidelines:
   - Keep sections logically organized
   - Use clear, descriptive headers
   - Maintain consistent formatting
   - Preserve technical terms exactly as written
   - Eliminate redundancy while keeping all unique information

{OUTPUT_FORMAT}

Remember:
- Let the content's natural structure guide organization
- Keep original terminology and key phrases intact
- Create clear hierarchical relationships
- Focus on making information easily retrievable
- Preserve all factual content while removing redundancy"""

# Constants for different scenarios
SINGLE_CHUNK_INFO = ""
CHUNKED_INFO = "chunk {CHUNK_NUMBER} of {TOTAL_CHUNKS} of the "

SINGLE_OUTPUT_FORMAT = """Format the output as a clean Markdown document:
<markdown_content>
# [Title]
[Overview]

## [Section]
- Points
</markdown_content>"""

CHUNK_OUTPUT_FORMAT = """For this chunk, format the output as:
<chunk_content>
<chunk>{CHUNK_NUMBER}</chunk>

[Markdown content following the same structure]

[{CHUNK_CONTINUATION}]

</chunk_content>

Also provide themes found in this chunk for maintaining consistency:
<themes>
{{
    "existing_sections": ["Sections found/continued in this chunk"],
    "current_themes": ["Main themes discussed"],
    "ongoing_topics": ["Topics that might continue in next chunks"],
    "terminology": ["Important terms and phrases to maintain"]
}}
</themes>"""

FIRST_CHUNK_CONTINUATION = "Continued in next chunk..."
MIDDLE_CHUNK_CONTINUATION = "Continued in next chunk..."
FINAL_CHUNK_CONTINUATION = ""


PROMPT_EXTRACT_BLOCKS_WITH_INSTRUCTION = """Here is the URL of the webpage:
<url>{URL}</url>

And here is the cleaned HTML content of that webpage:
<html>
{HTML}
</html>

Your task is to break down this HTML content into semantically relevant blocks, following the provided user's REQUEST, and for each block, generate a JSON object with the following keys:

- index: an integer representing the index of the block in the content
- content: a list of strings containing the text content of the block

This is the user's REQUEST, pay attention to it:
<request>
{REQUEST}
</request>

To generate the JSON objects:

1. Carefully read through the HTML content and identify logical breaks or shifts in the content that would warrant splitting it into separate blocks.

2. For each block:
   a. Assign it an index based on its order in the content.
   b. Analyze the content and generate ONE semantic tag that describe what the block is about.
   c. Extract the text content, EXACTLY SAME AS GIVE DATA, clean it up if needed, and store it as a list of strings in the "content" field.

3. Ensure that the order of the JSON objects matches the order of the blocks as they appear in the original HTML content.

4. Double-check that each JSON object includes all required keys (index, tag, content) and that the values are in the expected format (integer, list of strings, etc.).

5. Make sure the generated JSON is complete and parsable, with no errors or omissions.

6. Make sur to escape any special characters in the HTML content, and also single or double quote to avoid JSON parsing issues.

7. Never alter the extracted content, just copy and paste it as it is.

Please provide your output within <blocks> tags, like this:

<blocks>
[{
  "index": 0,
  "tags": ["introduction"],
  "content": ["This is the first paragraph of the article, which provides an introduction and overview of the main topic."]
},
{
  "index": 1,
  "tags": ["background"],
  "content": ["This is the second paragraph, which delves into the history and background of the topic.",
              "It provides context and sets the stage for the rest of the article."]
}]
</blocks>

**Make sure to follow the user instruction to extract blocks aligin with the instruction.**

Remember, the output should be a complete, parsable JSON wrapped in <blocks> tags, with no omissions or errors. The JSON objects should semantically break down the content into relevant blocks, maintaining the original order."""
