import os
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

st.set_page_config(page_title="🎬 YouTube Video Summarizer", layout="centered")

st.markdown("""
    <style>
    .output-box {
        background-color: #f9f9f9;
        padding: 20px;
        border-left: 6px solid #5b9bd5;
        border-radius: 10px;
        margin-top: 10px;
    }
    .transcript-box {
        background-color: #fefefe;
        padding: 15px;
        border: 1px solid #ddd;
        border-radius: 10px;
        max-height: 300px;
        overflow-y: scroll;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎥 YouTube Video Summarizer")
st.markdown("""
**Turn any YouTube video into a digestible summary.**

Paste a video URL below, and this app will extract its transcript and generate a concise summary.
""")

video_url = st.text_input("🔗 Enter YouTube Video URL")


def extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        match = re.search(r"youtu\.be/([^?\s]+)", url)
    elif "watch?v=" in url:
        match = re.search(r"watch\?v=([^&\s]+)", url)
    else:
        match = None
    return match.group(1) if match else None


def chunk_text(text: str, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def summarize_with_dynamic_params(text: str, summarizer, default_max: int, default_min: int) -> str:
    word_count = len(text.split())
    if word_count < default_min + 5:
        return text.strip()
    if word_count < default_max:
        max_length = word_count
        min_length = max(5, max_length // 2)
        if max_length <= min_length:
            max_length = min_length + 1
    else:
        max_length = default_max
        min_length = default_min

    result = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return result[0]['summary_text']


def summarize_long_text(text: str, summarizer, chunk_size=1000, overlap=100) -> str:
    chunks = chunk_text(text, chunk_size, overlap)
    partials = []
    for chunk in chunks:
        if len(chunk.split()) < 60:
            partials.append(chunk.strip())
        else:
            summarized = summarize_with_dynamic_params(chunk, summarizer, default_max=200, default_min=100)
            partials.append(summarized)
    return "\n\n".join(partials)


if st.button("✨ Summarize Video"):
    if video_url.strip():
        video_id = extract_video_id(video_url)
        if video_id:
            with st.spinner("Fetching transcript and summarizing..."):
                try:
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                    transcript_text = " ".join([segment["text"] for segment in transcript_data])

                    st.subheader("📃 Full Transcript")
                    st.markdown(f"<div class='transcript-box'>{transcript_text}</div>", unsafe_allow_html=True)

                    st.info("Transcript fetched. Starting summarization...")

                    # Use only the first 5000 characters for smoother performance
                    summarizer = pipeline("summarization")  # Use default model
                    short_transcript = transcript_text[:5000]
                    final_summary = summarize_long_text(short_transcript, summarizer)

                    st.subheader("📄 Summary")
                    st.markdown(f"<div class='output-box'>{final_summary}</div>", unsafe_allow_html=True)

                    st.success("Done! Here's your summary.")

                except Exception as e:
                    st.error(f"Error fetching transcript or summarizing: {e}")
        else:
            st.error("❌ Could not extract video ID from the URL. Please check the format.")
    else:
        st.warning("⚠️ Please enter a YouTube video URL.")
