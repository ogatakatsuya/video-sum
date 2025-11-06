"""
YouTube video download and video editing functionality using ffmpeg
"""

import subprocess
from pathlib import Path

import yt_dlp

from ..llm.transcription_analyzer import KeyPoint


def download_youtube_video(video_id, output_dir):
    """Download YouTube video and return file path"""
    youtube_url = f"https://www.youtube.com/watch?v={video_id}"
    output_path = Path(output_dir) / f"{video_id}.mp4"

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(output_path),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return output_path
    except Exception as e:
        raise Exception(f"Failed to download video: {str(e)}")


def create_highlight_video(video_path, key_points: list[KeyPoint], output_path):
    """Extract key sections using ffmpeg and merge into a single video"""
    temp_clips = []
    temp_dir = Path(video_path).parent
    concat_file = temp_dir / "concat_list.txt"

    try:
        # Extract each key section individually
        for i, kp in enumerate(key_points):
            clip_path = temp_dir / f"clip_{i}.mp4"
            temp_clips.append(clip_path)

            # Extract section using ffmpeg
            duration = kp.end_time - kp.start_time
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(kp.start_time),
                "-i",
                str(video_path),
                "-t",
                str(duration),
                "-c",
                "copy",
                str(clip_path),
            ]

            subprocess.run(cmd, check=True, capture_output=True)

        # Create clip list file
        concat_file = temp_dir / "concat_list.txt"
        with open(concat_file, "w") as f:
            for clip in temp_clips:
                f.write(f"file '{clip.absolute()}'\n")

        # Merge clips
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_path),
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        return output_path

    finally:
        # Delete temporary files
        for clip in temp_clips:
            if clip.exists():
                clip.unlink()
        if concat_file.exists():
            concat_file.unlink()
