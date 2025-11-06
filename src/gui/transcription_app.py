import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.llm.caption_analyzer import extract_key_events_from_captions
from src.util.file_utils import load_video_extensions
from src.util.video_utils import create_highlight_video, download_youtube_video
from text.transcription import download_youtube_audio, transcribe_audio

load_dotenv()

st.set_page_config(page_title="YouTube Video Player", page_icon="🎥", layout="centered")

st.title("🎥 YouTube Video Player")


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
        st.subheader("📝 Transcription")

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            st.warning("⚠️ OPENAI_API_KEY is not set in .env file")
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your OpenAI API key",
            )
        else:
            st.success("✅ API key loaded from .env")

        transcribe_button = st.button("🎤 Start Transcription", disabled=not api_key)

        if transcribe_button and api_key:
            with st.spinner("Downloading audio..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        st.info("Extracting audio from YouTube video...")
                        audio_path = download_youtube_audio(selected_video_id, temp_dir)

                        st.info("Transcribing audio...")
                        transcription = transcribe_audio(audio_path, api_key)

                        st.session_state.transcription = transcription

                        st.success("✅ Transcription completed!")

                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")

        if (
            hasattr(st.session_state, "transcription")
            and st.session_state.transcription
        ):
            transcription = st.session_state.transcription

            st.markdown("---")
            st.subheader("📄 Transcription Results")

            st.markdown("### 🎯 Key Point Extraction")

            gemini_api_key = os.getenv("GEMINI_API_KEY")

            if not gemini_api_key:
                st.warning("⚠️ GEMINI_API_KEY is not set in .env file")
                gemini_api_key = st.text_input(
                    "Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    help="Enter your Google Gemini API key",
                    key="gemini_key_input",
                )
            else:
                st.success("✅ Gemini API key loaded from .env")

            extract_button = st.button(
                "🎯 Extract Key Points", disabled=not gemini_api_key
            )

            if extract_button and gemini_api_key:
                with st.spinner("Extracting key points with Gemini..."):
                    try:
                        key_points_result = extract_key_events_from_captions(
                            transcription, gemini_api_key
                        )
                        st.session_state.key_points_result = key_points_result
                        st.success("✅ Key point extraction completed!")
                    except Exception as e:
                        st.error(f"❌ An error occurred: {str(e)}")

            if (
                hasattr(st.session_state, "key_points_result")
                and st.session_state.key_points_result
            ):
                key_points_result = st.session_state.key_points_result

                st.markdown("---")
                st.markdown("### 📌 Extracted Key Points")

                for i, kp in enumerate(key_points_result.key_points, 1):
                    start_min = int(kp.start_time // 60)
                    start_sec = int(kp.start_time % 60)
                    end_min = int(kp.end_time // 60)
                    end_sec = int(kp.end_time % 60)
                    time_str = (
                        f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                    )

                    with st.expander(f"**{i}. {kp.summary}**", expanded=True):
                        st.write(
                            f"⏱️ **Time**: {time_str} ({kp.end_time - kp.start_time:.1f}s)"
                        )
                        st.write(f"💡 **Reason**: {kp.reason}")

                st.markdown("---")
                st.markdown("### 🎬 Highlight Video Generation")

                generate_button = st.button(
                    "🎬 Generate Highlight Video", key="generate_highlight"
                )

                if generate_button:
                    with st.spinner("Downloading video..."):
                        try:
                            temp_dir = tempfile.mkdtemp()

                            st.info("Downloading YouTube video...")
                            video_path = download_youtube_video(
                                selected_video_id, temp_dir
                            )

                            st.info("Generating highlight video...")
                            output_path = (
                                Path(temp_dir) / f"{selected_video_id}_highlight.mp4"
                            )
                            create_highlight_video(
                                video_path, key_points_result.key_points, output_path
                            )

                            st.success("✅ Highlight video generation completed!")

                            with open(output_path, "rb") as video_file:
                                video_bytes = video_file.read()
                                st.download_button(
                                    label="📥 Download Highlight Video",
                                    data=video_bytes,
                                    file_name=f"{selected_video_id}_highlight.mp4",
                                    mime="video/mp4",
                                )

                            st.video(video_bytes)

                        except Exception as e:
                            st.error(f"❌ An error occurred: {str(e)}")
                        finally:
                            import shutil

                            if "temp_dir" in locals():
                                shutil.rmtree(temp_dir, ignore_errors=True)

            st.markdown("---")

            with st.expander("📖 Full Text", expanded=False):
                st.write(transcription.text)

            st.subheader("⏱️ Segments with Timestamps")

            if hasattr(transcription, "segments") and transcription.segments:
                for i, segment in enumerate(transcription.segments):
                    start_time = segment.start
                    end_time = segment.end

                    start_min = int(start_time // 60)
                    start_sec = int(start_time % 60)
                    end_min = int(end_time // 60)
                    end_sec = int(end_time % 60)

                    time_str = (
                        f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
                    )

                    with st.container():
                        col1, col2 = st.columns([1, 5])
                        with col1:
                            st.caption(time_str)
                        with col2:
                            st.write(segment.text)

                        if i < len(transcription.segments) - 1:
                            st.divider()

            with st.expander("🔧 JSON Output"):
                st.json(transcription.model_dump())

    else:
        st.error(f"❌ Video ID `{video_id_input}` not found")

        with st.expander("📋 Available Video IDs (first 20)"):
            sample_ids = list(filtered_videos.keys())[:20]
            for vid in sample_ids:
                st.code(f"{vid} ({filtered_videos[vid]})")

else:
    st.info("Please check extensions.json file")
