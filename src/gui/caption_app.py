import json
import os
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.llm.caption_analyzer import extract_key_events_from_captions
from src.text.caption import CaptionGenerator
from src.util.file_utils import load_video_extensions
from src.util.video_utils import download_youtube_video

load_dotenv()

st.set_page_config(
    page_title="Caption-based Video Summarization", page_icon="🎬", layout="centered"
)

st.title("🎬 Caption-based Video Summarization")
st.markdown("Select a YouTube video to generate captions and create a highlight video.")


def get_youtube_embed_html(video_id):
    """Generate YouTube video embed HTML (responsive)"""
    return f"""
    <style>
        .video-container {{
            position: relative;
            width: 100%;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            height: 0;
            overflow: hidden;
        }}
        .video-container iframe {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
    </style>
    <div class="video-container">
        <iframe
            src="https://www.youtube.com/embed/{video_id}"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>
    """


def parse_timestamp_to_seconds(timestamp: str) -> float:
    """Convert mm:ss.ff format to seconds"""
    try:
        parts = timestamp.split(":")
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    except Exception:
        return 0.0


def create_highlight_video_from_events(video_path, key_events, output_path):
    """Extract key sections using ffmpeg and merge into a single video"""
    temp_clips = []
    temp_dir = Path(video_path).parent
    concat_file = temp_dir / "concat_list.txt"

    try:
        # Extract each key section individually
        for i, event in enumerate(key_events):
            clip_path = temp_dir / f"clip_{i}.mp4"
            temp_clips.append(clip_path)

            # Extract section using ffmpeg
            duration = event.end_time - event.start_time
            cmd = [
                "ffmpeg",
                "-y",
                "-ss",
                str(event.start_time),
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


# Video selection from extensions.json
extensions = load_video_extensions()

if extensions:
    filtered_videos = extensions
else:
    st.warning("Failed to load video information")
    filtered_videos = {}

if filtered_videos:
    if "video_id" not in st.session_state:
        st.session_state.video_id = list(filtered_videos.keys())[0]

    col1, col2 = st.columns([3, 1])

    with col1:
        video_id_input = st.text_input(
            "Enter YouTube Video ID",
            value=st.session_state.video_id,
            placeholder="e.g., NFsc2pCdBAI",
        )
        if video_id_input != st.session_state.video_id:
            st.session_state.video_id = video_id_input

    with col2:
        st.write("")
        st.write("")
        if st.button("🔄 Random Select"):
            import random

            st.session_state.video_id = random.choice(list(filtered_videos.keys()))
            st.rerun()

    if video_id_input in filtered_videos:
        selected_video_id = video_id_input

        st.markdown("---")
        st.subheader("📹 Video Player")

        youtube_html = get_youtube_embed_html(selected_video_id)
        st.components.v1.html(youtube_html, height=410)

        st.markdown("---")
        st.subheader("📝 Caption Generation")

        model_path = st.text_input(
            "Caption Model Path",
            value="Qwen/Qwen3-VL-2B-Instruct",
            help="Hugging Face model path for caption generation",
        )

        generate_caption_button = st.button("🎬 Generate Captions")

        if generate_caption_button:
            with st.spinner("Generating captions..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        st.info("Downloading YouTube video...")
                        video_path = download_youtube_video(selected_video_id, temp_dir)

                        st.info("Generating captions with vision model...")
                        caption_generator = CaptionGenerator(model_path)
                        caption_text = caption_generator.generate_caption(
                            str(video_path)
                        )
                        st.session_state.caption_text = caption_text
                        st.session_state.selected_video_id = selected_video_id

                        st.success("Captions generated successfully!")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    import traceback

                    st.error(traceback.format_exc())

        # Display captions
        if hasattr(st.session_state, "caption_text") and st.session_state.caption_text:
            caption_text = st.session_state.caption_text

            st.markdown("---")
            st.subheader("📄 Generated Captions")

            # Display raw caption text
            with st.expander("📖 Raw Caption Output", expanded=True):
                st.code(caption_text, language="json")

            # Parse and display events
            try:
                caption_data = json.loads(caption_text)

                # Handle both array and object with events array
                if isinstance(caption_data, list):
                    events = caption_data
                elif isinstance(caption_data, dict) and "events" in caption_data:
                    events = caption_data["events"]
                else:
                    events = []

                if events:
                    st.markdown("### 🎯 Detected Events")
                    for i, event in enumerate(events, 1):
                        start = event.get("start_time", event.get("start", "N/A"))
                        end = event.get("end_time", event.get("end", "N/A"))
                        desc = event.get("description", event.get("caption", "N/A"))

                        with st.expander(
                            f"**Event {i}: {start} - {end}**", expanded=False
                        ):
                            st.write(f"**Time**: {start} - {end}")
                            st.write(f"**Description**: {desc}")

            except json.JSONDecodeError:
                st.warning("Could not parse caption as JSON. Displaying raw text.")

            # Key Event Extraction with Gemini
            st.markdown("---")
            st.markdown("### 🎯 Key Event Extraction")

            gemini_api_key = os.getenv("GEMINI_API_KEY")

            if not gemini_api_key:
                st.warning("GEMINI_API_KEY is not set in .env file")
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    help="Enter your Google Gemini API key",
                    key="gemini_key_input",
                )
            else:
                st.success("Gemini API key loaded from .env")

            extract_button = st.button(
                "🎯 Extract Key Events", disabled=not gemini_api_key
            )

            if extract_button and gemini_api_key:
                with st.spinner("Extracting key events with Gemini..."):
                    try:
                        key_events_result = extract_key_events_from_captions(
                            caption_text, gemini_api_key
                        )
                        st.session_state.key_events_result = key_events_result
                        st.success("Key event extraction completed!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        import traceback

                        st.error(traceback.format_exc())

            if (
                hasattr(st.session_state, "key_events_result")
                and st.session_state.key_events_result
            ):
                key_events_result = st.session_state.key_events_result

                st.markdown("---")
                st.markdown("### 📌 Extracted Key Events")

                for i, event in enumerate(key_events_result.key_events, 1):
                    start_min = int(event.start_time // 60)
                    start_sec = int(event.start_time % 60)
                    end_min = int(event.end_time // 60)
                    end_sec = int(event.end_time % 60)
                    time_str = (
                        f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                    )

                    with st.expander(f"**{i}. {event.summary}**", expanded=True):
                        st.write(
                            f"**Time**: {time_str} ({event.end_time - event.start_time:.1f}s)"
                        )
                        st.write(f"**Reason**: {event.reason}")

                # Highlight Video Generation
                st.markdown("---")
                st.markdown("### 🎬 Highlight Video Generation")

                generate_button = st.button(
                    "🎬 Generate Highlight Video", key="generate_highlight"
                )

                if generate_button:
                    with st.spinner("Generating highlight video..."):
                        try:
                            temp_dir = tempfile.mkdtemp()

                            # Download video from YouTube using stored video_id
                            st.info("Downloading YouTube video...")
                            video_path = download_youtube_video(
                                st.session_state.selected_video_id, temp_dir
                            )

                            st.info("Generating highlight video...")
                            output_path = (
                                Path(temp_dir)
                                / f"{st.session_state.selected_video_id}_highlight.mp4"
                            )
                            create_highlight_video_from_events(
                                video_path, key_events_result.key_events, output_path
                            )

                            st.success("Highlight video generation completed!")

                            with open(output_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                st.download_button(
                                    label="📥 Download Highlight Video",
                                    data=video_bytes,
                                    file_name=f"{st.session_state.selected_video_id}_highlight.mp4",
                                    mime="video/mp4",
                                )

                            st.video(video_bytes)

                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                            import traceback

                            st.error(traceback.format_exc())
                        finally:
                            import shutil

                            if "temp_dir" in locals() and Path(temp_dir).exists():
                                shutil.rmtree(temp_dir, ignore_errors=True)

    else:
        st.error(f"Video ID `{video_id_input}` not found")

        with st.expander("📋 Available Video IDs (first 20)"):
            sample_ids = list(filtered_videos.keys())[:20]
            for vid in sample_ids:
                st.code(f"{vid} ({filtered_videos[vid]})")

else:
    st.info("Please check extensions.json file")
