"""
Text processing utilities for cleaning and parsing model outputs.
"""

import json
from typing import Any, Union


def clean_markdown_code_blocks(text: str) -> str:
    """
    Remove markdown code blocks from text.
    
    Args:
        text (str): Input text that may contain markdown code blocks
        
    Returns:
        str: Cleaned text without markdown code blocks
    """
    clean_text = text.strip()
    
    # Remove opening code blocks
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]  # Remove ```json
    elif clean_text.startswith("```"):
        clean_text = clean_text[3:]   # Remove ```
    
    # Remove closing code blocks
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]  # Remove trailing ```
    
    return clean_text.strip()


def parse_json_with_cleanup(text: str) -> Union[dict[str, Any], list[Any]]:
    """
    Parse JSON text after cleaning markdown code blocks.
    
    Args:
        text (str): Input text that may contain JSON with markdown code blocks
        
    Returns:
        Union[Dict, List]: Parsed JSON object
        
    Raises:
        json.JSONDecodeError: If the text cannot be parsed as JSON
    """
    cleaned_text = clean_markdown_code_blocks(text)
    return json.loads(cleaned_text)


def extract_events_from_caption_data(caption_data: Union[str, dict[str, Any], list[Any]]) -> list[dict[str, Any]]:
    """
    Extract events from caption data, handling various formats.
    
    Args:
        caption_data: Caption data as string, dict, or list
        
    Returns:
        List[Dict]: List of event dictionaries
    """
    # If it's a string, try to parse as JSON
    if isinstance(caption_data, str):
        try:
            caption_data = parse_json_with_cleanup(caption_data)
        except json.JSONDecodeError:
            return []
    
    # Handle different data structures
    if isinstance(caption_data, list):
        return caption_data
    elif isinstance(caption_data, dict) and "events" in caption_data:
        return caption_data["events"]
    else:
        return []
