"""
Prompt templates for LLM interactions.
"""


class PromptTemplates:
    """Collection of prompt templates for knowledge extraction."""
    
    SYSTEM_PROMPT = """You are a helpful assistant that extracts and organizes knowledge from social media content. 
Your task is to:
1. Understand the content (which may be in Hindi, English, or Hinglish)
2. Extract key information and insights
3. Structure the knowledge in a clear, useful format
4. Respond in the same language as the input (or English if mixed)

Be concise but comprehensive. Focus on actionable insights and memorable facts."""

    SUMMARIZE = """Summarize the following social media content into a clear, concise summary.
Focus on the main points and key takeaways.

Content:
{content}

Provide a summary in 2-3 paragraphs."""

    EXTRACT_KEY_POINTS = """Extract the key points from the following social media content.
List them as bullet points, focusing on:
- Main ideas or claims
- Actionable advice or tips
- Important facts or statistics
- Notable quotes or statements

Content:
{content}

Format the output as a bullet-point list."""

    STRUCTURE_KNOWLEDGE = """Analyze the following social media content and structure it as organized knowledge.

Content:
{content}

Provide the output in the following JSON format:
{{
    "title": "A descriptive title for this content",
    "summary": "A 2-3 sentence summary",
    "key_points": ["point 1", "point 2", ...],
    "topics": ["topic1", "topic2", ...],
    "actionable_items": ["action 1", "action 2", ...],
    "quotes": ["notable quote 1", ...],
    "language": "detected language (en/hi/hinglish)"
}}

Ensure the JSON is valid and complete."""

    CATEGORIZE = """Categorize the following content into appropriate topics.

Content:
{content}

Available categories:
- Technology
- Health & Fitness
- Finance & Money
- Productivity
- Self-Improvement
- Science
- Current Events
- Entertainment
- Education
- Lifestyle
- Other

Return a comma-separated list of the most relevant categories (1-3)."""

    CLEAN_TRANSCRIPT = """The following is an audio transcription that may contain errors or unclear speech.
Clean it up while preserving the original meaning and language:
- Fix obvious transcription errors
- Add proper punctuation
- Keep the original language (Hindi/English/Hinglish)
- Preserve the speaker's intent

Raw transcription:
{content}

Provide the cleaned version:"""

    COMBINE_CONTENT = """The following is combined content from a social media post including:
- Caption text
- Extracted text from images (OCR)
- Audio transcription (if video)
- Top comments

Analyze all of this together and provide a comprehensive understanding:

Caption:
{caption}

OCR Text (from images):
{ocr_text}

Transcription (from video/audio):
{transcription}

Top Comments:
{comments}

Provide:
1. A unified summary of all content
2. Key insights and takeaways
3. Any disagreements or additions from comments"""

    @classmethod
    def get_summarize_prompt(cls, content: str) -> str:
        """Get the summarization prompt."""
        return cls.SUMMARIZE.format(content=content)
    
    @classmethod
    def get_key_points_prompt(cls, content: str) -> str:
        """Get the key points extraction prompt."""
        return cls.EXTRACT_KEY_POINTS.format(content=content)
    
    @classmethod
    def get_structure_prompt(cls, content: str) -> str:
        """Get the knowledge structuring prompt."""
        return cls.STRUCTURE_KNOWLEDGE.format(content=content)
    
    @classmethod
    def get_categorize_prompt(cls, content: str) -> str:
        """Get the categorization prompt."""
        return cls.CATEGORIZE.format(content=content)
    
    @classmethod
    def get_clean_transcript_prompt(cls, content: str) -> str:
        """Get the transcript cleaning prompt."""
        return cls.CLEAN_TRANSCRIPT.format(content=content)
    
    @classmethod
    def get_combined_prompt(
        cls, 
        caption: str = "",
        ocr_text: str = "",
        transcription: str = "",
        comments: str = ""
    ) -> str:
        """Get the combined content analysis prompt."""
        return cls.COMBINE_CONTENT.format(
            caption=caption or "(no caption)",
            ocr_text=ocr_text or "(no OCR text)",
            transcription=transcription or "(no transcription)",
            comments=comments or "(no comments)"
        )
