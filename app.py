import os
import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi

# Make sure to call set_page_config() as the first Streamlit command.
st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")

try:
    from transformers import pipeline
except ImportError as e:
    st.error("Error importing pipeline from transformers. Please run 'pip install --upgrade transformers'")
    raise e

# Suppress unnecessary warnings.
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ----- Custom CSS for Enhanced Interface -----
custom_css = """
<style>
/* Overall page background and layout adjustments */
body {
    background-color: #f2f2f2;
    font-family: 'Segoe UI', sans-serif;
}

/* Title style */
h1 {
    color: #333333;
    text-align: center;
}

/* Subtitle style */
h2, h3 {
    color: #444444;
}

/* Container for main content with a box shadow and rounded corners */
.main-container {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
}

/* Styling input fields and buttons */
input[type="text"] {
    border-radius: 5px;
    border: 1px solid #cccccc;
    padding: 8px;
}

.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 8px 16px;
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #45a049;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f9f9f9;
    border-right: 1px solid #e6e6e6;
}

/* Expander style for transcript */
[data-testid="stExpander"] {
    background-color: #fafafa;
    border-radius: 5px;
    border: 1px solid #e6e6e6;
}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# ----- Sidebar Instructions -----
st.sidebar.title("About This Tool")
st.sidebar.info(
    """
    This application generates a detailed summary of a YouTube video's transcript.
    
    **Instructions:**
    1. Enter the URL of the YouTube video.
    2. Click **Summarize Video**.
    3. The full transcript will appear in an expandable section.
    4. A detailed summary will display below.
    """
)

# ----- Main Container -----
with st.container():
    st.title("🎥 YouTube Video Summarizer")
    st.write("Generate a detailed summary of any YouTube video transcript.")

    def extract_video_id(url: str) -> str:
        """
        Extract the video ID from common YouTube URL formats.
        """
        if "youtu.be/" in url:
            match = re.search(r"youtu\.be/([^?\s]+)", url)
        elif "watch?v=" in url:
            match = re.search(r"watch\?v=([^&\s]+)", url)
        else:
            match = None
        return match.group(1) if match else None

    #############################################
    # Summarization Functions (Token-based Chunking)
    #############################################
    def chunk_text_by_tokens(text: str, tokenizer, max_tokens=1024, overlap_tokens=100):
        """
        Splits the text into chunks that are at most `max_tokens` in length.
        Chunks overlap by `overlap_tokens` to preserve context.
        """
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        start = 0
        while start < len(token_ids):
            end = start + max_tokens
            chunk_ids = token_ids[start:end]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            start = end - overlap_tokens  # move start back for overlap
        return chunks

    def generate_summary(text: str, summarizer):
        """
        Generates a detailed summary for long text by:
          1. Splitting the text into token-based chunks.
          2. Summarizing each chunk individually.
          3. Concatenating all partial summaries.
          
        No final aggressive summarization pass is performed to preserve details.
        """
        tokenizer = summarizer.tokenizer
        # Use 1024 tokens as the maximum chunk size (or the model's limit if lower).
        max_tokens = min(tokenizer.model_max_length, 1024)
        overlap_tokens = 100
        chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        partial_summaries = []
        st.info(f"Processing {len(chunks)} chunk(s) for summarization...")

        # Process each chunk independently.
        for idx, chunk in enumerate(chunks):
            token_count = len(tokenizer.encode(chunk, add_special_tokens=False))
            if token_count < 60:
                partial_summaries.append(chunk.strip())
            else:
                try:
                    # Adjust parameters to produce a detailed summary per chunk.
                    result = summarizer(
                        chunk,
                        max_length=250,  # Adjust these values if needed.
                        min_length=180,
                        do_sample=False,
                        truncation=True
                    )
                    partial_summaries.append(result[0]['summary_text'])
                except Exception as e:
                    st.error(f"Error summarizing chunk {idx+1}: {e}")
                    partial_summaries.append(chunk.strip())

        # Combine all partial summaries.
        final_summary = "\n\n".join(partial_summaries)
        return final_summary

    #############################################
    # Main Application: Summarize Video
    #############################################
    video_url = st.text_input("Enter YouTube Video URL:")
    if st.button("Summarize Video"):
        if video_url.strip():
            video_id = extract_video_id(video_url)
            if video_id:
                try:
                    with st.spinner("Fetching transcript..."):
                        transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                        transcript_text = " ".join([segment["text"] for segment in transcript_data])
                    
                    with st.expander("📃 View Full Transcript"):
                        st.write(transcript_text)

                    with st.spinner("Generating summary..."):
                        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                        final_summary = generate_summary(transcript_text, summarizer)
                    
                    st.subheader("📄 Detailed Summary")
                    st.write(final_summary)
                except Exception as e:
                    st.error(f"Error fetching transcript or summarizing: {e}")
            else:
                st.error("Could not extract video ID from the provided URL. Please check the format.")
        else:
            st.warning("Please enter a YouTube video URL.")
