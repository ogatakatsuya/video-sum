"""
Caption-based video analysis functionality using Gemini API
Extracts key events from caption data with timestamps
"""

import json

from google import genai
from pydantic import BaseModel


# Pydantic model definitions
class CaptionEvent(BaseModel):
    """Video event with caption and timestamp"""

    start_time: str  # Start time in mm:ss.ff format
    end_time: str  # End time in mm:ss.ff format
    description: str  # Description of the event


class KeyEvent(BaseModel):
    """Key event selected from captions"""

    summary: str  # Summary of the event
    start_time: float  # Start time (seconds)
    end_time: float  # End time (seconds)
    reason: str  # Why it's important


class KeyEventsResult(BaseModel):
    """List of key events"""

    key_events: list[KeyEvent]


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Convert mm:ss.ff format to seconds"""
    try:
        parts = timestamp.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except Exception:
        return 0.0


def extract_key_events_from_captions(caption_json: str, api_key: str):
    """Extract key events from caption data using Gemini 2.5 Flash"""
    client = genai.Client(api_key=api_key)

    # Parse caption JSON
    try:
        caption_data = json.loads(caption_json)

        # Handle both array and object with events array
        if isinstance(caption_data, list):
            events = caption_data
        elif isinstance(caption_data, dict) and "events" in caption_data:
            events = caption_data["events"]
        else:
            raise ValueError("Invalid caption format")

    except Exception as e:
        raise ValueError(f"Failed to parse caption JSON: {str(e)}")

    # Format events for Gemini
    events_text = ""
    for i, event in enumerate(events, 1):
        start = event.get("start_time", event.get("start", ""))
        end = event.get("end_time", event.get("end", ""))
        desc = event.get("description", event.get("caption", ""))
        events_text += f"{i}. [{start} - {end}] {desc}\n"

    prompt = f"""
Below is a list of events extracted from a video with their timestamps and descriptions.
Extract 3-5 most important and interesting events from this list.

Requirements:
1. Select 3-5 most important, interesting, or informative events
2. Order them by importance (most important first)
3. Explain why each event is important or interesting
4. Convert timestamps from mm:ss.ff format to seconds for start_time and end_time
5. Provide a concise summary for each selected event

Video Events:
{events_text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": KeyEventsResult,
        },
    )

    return response.parsed
