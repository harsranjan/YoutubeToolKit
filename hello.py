import os
import re
import io
import tempfile
import zipfile

import streamlit as st
import nltk
import spacy
import cv2
import yt_dlp

from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline, T5Tokenizer
from fpdf import FPDF
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# ─────────────────────────────────────────────────────────────────────────────
# Environment & Model Setup
# ─────────────────────────────────────────────────────────────────────────────
os.environ["STREAMLIT_SERVER_FILEWATCHER_TYPE"] = "none"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# download punkt for NLTK sentence splitting
nltk.download("punkt", quiet=True)
# load spaCy model
nlp = spacy.load("en_core_web_sm")

# Quiz & Notes pipelines — force CPU with device=-1
tokenizer_qg = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl", use_fast=False)
qg_pipeline = pipeline(
    "text2text-generation",
    model="valhalla/t5-small-qg-hl",
    tokenizer=tokenizer_qg,
    device=-1
)
summarizer_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1,
    use_fast=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Video Summarizer Helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_video_id(url: str) -> str:
    if "youtu.be/" in url:
        m = re.search(r"youtu\.be/([^?\s]+)", url)
    elif "watch?v=" in url:
        m = re.search(r"watch\?v=([^&\s]+)", url)
    else:
        m = None
    return m.group(1) if m else None

def chunk_text_by_tokens(text: str, tokenizer, max_tokens=1024, overlap_tokens=100):
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks, start = [], 0
    while start < len(token_ids):
        end = start + max_tokens
        chunk = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)
        chunks.append(chunk)
        start = end - overlap_tokens
    return chunks

def generate_summary(text: str, summarizer):
    tokenizer = summarizer.tokenizer
    max_tokens = min(tokenizer.model_max_length, 1024)
    chunks = chunk_text_by_tokens(text, tokenizer, max_tokens=max_tokens, overlap_tokens=100)
    partials = []
    st.info(f"Processing {len(chunks)} chunk(s)...")
    for idx, chunk in enumerate(chunks, start=1):
        count = len(tokenizer.encode(chunk, add_special_tokens=False))
        if count < 60:
            partials.append(chunk.strip())
        else:
            try:
                out = summarizer(chunk,
                                 max_length=250,
                                 min_length=180,
                                 do_sample=False,
                                 truncation=True)
                partials.append(out[0]["summary_text"])
            except Exception as e:
                st.error(f"Chunk {idx} error: {e}")
                partials.append(chunk.strip())
    return "\n\n".join(partials)

# ─────────────────────────────────────────────────────────────────────────────
# Quiz & Notes Helpers
# ─────────────────────────────────────────────────────────────────────────────
def generate_flashcards(summary_text, min_words=5, max_q=50):
    cards = []
    doc = nlp(summary_text)
    for sent in doc.sents:
        if len(cards) >= max_q:
            break
        s = sent.text.strip()
        if len(s.split()) < min_words:
            continue
        # choose a highlight candidate
        if sent.ents:
            cand = sent.ents[0].text.strip()
        else:
            nc = list(sent.noun_chunks)
            cand = nc[0].text.strip() if nc else s.split()[0]
        # highlight it
        if cand in s:
            inp = s.replace(cand, f"<hl> {cand} <hl>", 1)
        else:
            inp = s
        # generate
        try:
            out = qg_pipeline(f"generate question: {inp}")
            txt = out[0]["generated_text"].strip()
            if "Q:" in txt and "A:" in txt:
                q = txt.split("A:")[0].replace("Q:", "").strip()
                a = txt.split("A:")[1].strip()
                if q and a:
                    cards.append({"question": q, "answer": s})
                    continue
            if txt:
                cards.append({"question": txt, "answer": s})
        except Exception as e:
            st.error(f"Flashcard error: {e}")
    return cards

def generate_study_notes(summary_text):
    try:
        out = summarizer_pipeline(summary_text,
                                  max_length=200,
                                  min_length=50,
                                  do_sample=False,
                                  truncation=True)
        return out[0].get("summary_text", "No notes generated.")
    except Exception as e:
        return f"Error: {e}"

# ─────────────────────────────────────────────────────────────────────────────
# Frame Extractor & PDF Helpers
# ─────────────────────────────────────────────────────────────────────────────
def download_video(url: str, filename: str, max_retries=3) -> str:
    opts = {"outtmpl": filename, "format": "best"}
    for i in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            return filename
        except yt_dlp.utils.DownloadError as e:
            st.warning(f"Download failed ({i+1}/{max_retries}): {e}")
    raise Exception("Download failed after retries.")

def get_video_id_for_frames(url: str) -> str:
    for patt in [r"shorts/(\w+)", r"youtu\.be/([\w\-_]+)", r"v=([\w\-_]+)", r"live/(\w+)"]:
        m = re.search(patt, url)
        if m:
            return m.group(1)
    return None

def extract_unique_frames(video_file, tmp_dir, interval=3, threshold=0.8):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    last_gray = None
    saved = None
    saved_no = -1
    frame_no = 0
    collected = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_no % interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (128, 72))
            if last_gray is None:
                path = os.path.join(tmp_dir, f"frame{frame_no:04d}_{frame_no//fps}.png")
                cv2.imwrite(path, frame)
                collected.append((path, frame_no // fps))
                saved = frame
                saved_no = frame_no
            else:
                sim = ssim(gray, last_gray, data_range=gray.max()-gray.min())
                if sim < threshold and (frame_no - saved_no) > fps:
                    path = os.path.join(tmp_dir, f"frame{frame_no:04d}_{frame_no//fps}.png")
                    cv2.imwrite(path, saved)
                    collected.append((path, frame_no // fps))
                    saved_no = frame_no
                saved = frame
            last_gray = gray
        frame_no += 1

    cap.release()
    return collected

def convert_frames_to_pdf_bytes(frames_and_ts):
    pdf = FPDF("L", "pt", "A4")
    pdf.set_auto_page_break(False)
    w, h = pdf.w, pdf.h
    for img_path, ts in frames_and_ts:
        img = Image.open(img_path)
        iw, ih = img.size
        scale = min(w/iw, h/ih)
        nw, nh = iw*scale, ih*scale
        xoff, yoff = (w-nw)/2, (h-nh)/2

        pdf.add_page()
        pdf.image(img_path, x=xoff, y=yoff, w=nw, h=nh)

        hh, mm, ss = ts//3600, (ts%3600)//60, ts%60
        stamp = f"{hh:02d}:{mm:02d}:{ss:02d}"
        region = img.crop((0,0,60,20)).convert("L").resize((1,1)).getpixel((0,0))
        pdf.set_text_color(255,255,255) if region<64 else pdf.set_text_color(0,0,0)
        pdf.set_xy(10,10)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 0, stamp)

    return pdf.output(dest="S").encode("latin1")

def get_video_title(url: str) -> str:
    with yt_dlp.YoutubeDL({"skip_download": True, "ignoreerrors": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title", "video")
    for c in '/\\:*?"<>|':
        title = title.replace(c, "-")
    return title.strip(".")

def process_single_video_to_bytes(url: str):
    tmp_video = "temp.mp4"
    download_video(url, tmp_video)
    title = get_video_title(url)
    pdf_name = f"{title}.pdf"

    with tempfile.TemporaryDirectory() as td:
        frames = extract_unique_frames(tmp_video, td)
        pdf_bytes = convert_frames_to_pdf_bytes(frames)

    os.remove(tmp_video)
    return pdf_bytes, pdf_name

def process_playlist_to_zip(url: str):
    with yt_dlp.YoutubeDL({"ignoreerrors": True, "playlistend": 1000, "extract_flat": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    entries = info.get("entries", [])
    if not entries:
        raise Exception("No videos in playlist.")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, e in enumerate(entries, start=1):
            video_url = e.get("url")
            st.write(f"Processing {i}/{len(entries)}: {video_url}")
            pdf_bytes, name = process_single_video_to_bytes(video_url)
            zf.writestr(name, pdf_bytes)
    buf.seek(0)
    return buf

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit App: Combined UI
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="YouTube Multi-Tool", layout="centered")
st.sidebar.title("YouTube Toolkit")

choice = st.sidebar.radio(
    "Choose feature",
    ["Video Summarizer", "Quiz & Notes Generator", "Frame Extractor & PDF Generator"]
)

if choice == "Video Summarizer":
    st.header("🎥 YouTube Video Summarizer")
    st.write("Enter a YouTube URL to fetch its transcript and generate a detailed summary.")
    url = st.text_input("YouTube Video URL")
    if st.button("Summarize Video"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            vid = extract_video_id(url)
            if not vid:
                st.error("Could not extract video ID.")
            else:
                try:
                    with st.spinner("Fetching transcript..."):
                        data = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
                        full_txt = " ".join([seg["text"] for seg in data])
                    with st.expander("View Full Transcript"):
                        st.write(full_txt)
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(full_txt, summarizer_pipeline)
                    st.subheader("Detailed Summary")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Error: {e}")

elif choice == "Quiz & Notes Generator":
    st.header("📖 YouTube Quiz & Notes Generator")
    st.write("Paste your transcript summary below to auto-generate flashcards and study notes.")
    txt = st.text_area("Transcript Summary", height=200)
    if st.button("Generate Quiz & Study Notes"):
        if not txt.strip():
            st.error("Please paste a summary.")
        else:
            with st.spinner("Generating flashcards..."):
                cards = generate_flashcards(txt)
            with st.spinner("Generating study notes..."):
                notes = generate_study_notes(txt)
            st.markdown("---")
            st.subheader("Flashcards")
            if cards:
                for i, c in enumerate(cards, 1):
                    st.markdown(f"**Q{i}:** {c['question']}")
                    with st.expander("Show Answer"):
                        st.write(c["answer"])
            else:
                st.info("No flashcards generated.")
            st.markdown("---")
            st.subheader("Study Notes")
            st.write(notes)

else:  # Frame Extractor & PDF
    st.header("🖼️ YouTube Frame Extractor & PDF Generator")
    st.write("Download a video or playlist, extract unique frames, and package them into PDF(s).")
    url = st.text_input("YouTube Video or Playlist URL")
    if st.button("Process"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            vid = get_video_id_for_frames(url)
            with st.spinner("Processing..."):
                try:
                    if vid:
                        pdf_bytes, name = process_single_video_to_bytes(url)
                        st.success("Done! Download below:")
                        st.download_button("Download PDF", data=pdf_bytes, file_name=name, mime="application/pdf")
                    else:
                        zip_buf = process_playlist_to_zip(url)
                        st.success("Playlist complete! Download below:")
                        st.download_button("Download All PDFs (ZIP)",
                                           data=zip_buf,
                                           file_name="extracted_frames.zip",
                                           mime="application/zip")
                except Exception as e:
                    st.error(f"Error: {e}")
