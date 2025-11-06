"""
Transcription text analysis functionality using Gemini API
Extracts key points with timestamps
"""

from google import genai
from pydantic import BaseModel


# Pydantic model definitions
class KeyPoint(BaseModel):
    """Key point with its timestamp"""

    summary: str  # Summary of the point
    start_time: float  # Start time (seconds)
    end_time: float  # End time (seconds)
    reason: str  # Why it's important


class KeyPointsResult(BaseModel):
    """List of key points"""

    key_points: list[KeyPoint]


def extract_key_points_with_timestamps(transcription, api_key):
    """Extract key points with timestamps from transcription using Gemini 2.5 Flash"""
    client = genai.Client(api_key=api_key)

    # Format segment information
    segments_text = ""
    for segment in transcription.segments:
        segments_text += f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n"

    prompt = f"""
Below is the transcription text of a video. Each line contains [start_time - end_time] and the text.
Extract 3-5 most important sections from this text and specify their start and end times.

Requirements:
1. Select 3-5 most important and valuable sections
2. Each section should be a meaningful unit (approximately 10-60 seconds)
3. Order them by importance (most important first)
4. Explain why each section is important
5. Start and end times must be accurately selected from the original segment times

Transcription text:
{segments_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": KeyPointsResult,
        },
    )

    return response.parsed
