"""
File loading utility
"""

import json
from pathlib import Path


def load_video_extensions(json_path="data/extensions.json"):
    """Load extensions.json and get video ID to extension mapping"""
    json_file = Path(json_path)

    if not json_file.exists():
        return {}

    with open(json_file, "r") as f:
        return json.load(f)


def load_local_videos(video_dir="data/"):
    """Load local video files from specified directory

    Args:
        video_dir: Directory path to search for video files

    Returns:
        Dictionary mapping video filename to full path
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]
    video_dir_path = Path(video_dir)

    if not video_dir_path.exists():
        return {}

    videos = {}
    for ext in video_extensions:
        for video_file in video_dir_path.glob(f"*{ext}"):
            videos[video_file.name] = str(video_file.absolute())

    return videos
