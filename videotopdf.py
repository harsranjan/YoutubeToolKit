import streamlit as st
import tempfile
import os
import re
import io
import zipfile
import cv2
from fpdf import FPDF
from PIL import Image, ImageFile
import yt_dlp
from skimage.metrics import structural_similarity as ssim
from scipy.spatial import distance

# Ensure that PIL's ImageFile module is properly registered
import sys
sys.modules['ImageFile'] = ImageFile

# ----------------------------------
# Helper Functions (mostly from your original script)
# ----------------------------------

def download_video(url, filename, max_retries=3):
    ydl_opts = {
        'outtmpl': filename,
        'format': 'best',
    }
    retries = 0
    while retries < max_retries:
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                return filename
        except yt_dlp.utils.DownloadError as e:
            st.warning(f"Error downloading video: {e}. Retrying... (Attempt {retries + 1}/{max_retries})")
            retries += 1
    raise Exception("Failed to download video after multiple attempts.")

def get_video_id(url):
    # Match YouTube Shorts URLs
    video_id_match = re.search(r"shorts\/(\w+)", url)
    if video_id_match:
        return video_id_match.group(1)

    # Match youtube.be shortened URLs
    video_id_match = re.search(r"youtu\.be\/([\w\-_]+)(\?.*)?", url)
    if video_id_match:
        return video_id_match.group(1)
               
    # Match regular YouTube URLs
    video_id_match = re.search(r"v=([\w\-_]+)", url)
    if video_id_match:
        return video_id_match.group(1)

    # Match YouTube live stream URLs
    video_id_match = re.search(r"live\/(\w+)", url)  
    if video_id_match:
        return video_id_match.group(1)

    return None

def get_playlist_videos(playlist_url):
    ydl_opts = {
        'ignoreerrors': True,
        'playlistend': 1000,  # Maximum number of videos to fetch
        'extract_flat': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_info = ydl.extract_info(playlist_url, download=False)
        return [entry['url'] for entry in playlist_info['entries'] if entry]

def extract_unique_frames(video_file, output_folder, n=3, ssim_threshold=0.8):
    cap = cv2.VideoCapture(video_file)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    last_frame = None
    saved_frame = None
    frame_number = 0
    last_saved_frame_number = -1
    timestamps = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % n == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.resize(gray_frame, (128, 72))

            if last_frame is not None:
                similarity = ssim(gray_frame, last_frame, data_range=gray_frame.max() - gray_frame.min())
                if similarity < ssim_threshold:
                    if saved_frame is not None and frame_number - last_saved_frame_number > fps:
                        frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{frame_number // fps}.png')
                        cv2.imwrite(frame_path, saved_frame)
                        timestamps.append((frame_number, frame_number // fps))
                    saved_frame = frame
                    last_saved_frame_number = frame_number
                else:
                    saved_frame = frame
            else:
                frame_path = os.path.join(output_folder, f'frame{frame_number:04d}_{frame_number // fps}.png')
                cv2.imwrite(frame_path, frame)
                timestamps.append((frame_number, frame_number // fps))
                last_saved_frame_number = frame_number

            last_frame = gray_frame

        frame_number += 1

    cap.release()
    return timestamps

def convert_frames_to_pdf(input_folder, output_file, timestamps):
    from PIL import Image
    # Sorted list of image files based on frame numbers in filenames
    frame_files = sorted(os.listdir(input_folder), key=lambda x: int(x.split('_')[0].split('frame')[-1]))
    
    pdf = FPDF("L", unit="pt", format="A4")
    pdf.set_auto_page_break(False)
    page_width, page_height = pdf.w, pdf.h

    for frame_file, (frame_number, timestamp_seconds) in zip(frame_files, timestamps):
        frame_path = os.path.join(input_folder, frame_file)
        image = Image.open(frame_path)
        img_width, img_height = image.size

        # Scale image proportionally to fit on the page
        scale = min(page_width / img_width, page_height / img_height)
        new_width = img_width * scale
        new_height = img_height * scale

        # Center the image on the page
        x_offset = (page_width - new_width) / 2
        y_offset = (page_height - new_height) / 2

        pdf.add_page()
        pdf.image(frame_path, x=x_offset, y=y_offset, w=new_width, h=new_height)

        # Format the timestamp as HH:MM:SS
        timestamp = f"{timestamp_seconds // 3600:02d}:{(timestamp_seconds % 3600) // 60:02d}:{timestamp_seconds % 60:02d}"

        # Choose text color based on image brightness
        text_region = image.crop((10, 10, 70, 25)).convert("L")
        mean_pixel_value = text_region.resize((1, 1)).getpixel((0, 0))
        if mean_pixel_value < 64:
            pdf.set_text_color(255, 255, 255)
        else:
            pdf.set_text_color(0, 0, 0)

        pdf.set_xy(10, 10)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 0, timestamp)

    pdf.output(output_file)
    st.info(f"PDF saved as: {output_file}")

def get_video_title(url):
    ydl_opts = {
        'skip_download': True,
        'ignoreerrors': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_info = ydl.extract_info(url, download=False)
        title = video_info['title']
        # Sanitize file name
        invalid_chars = '/\\:*?"<>|'
        for char in invalid_chars:
            title = title.replace(char, '-')
        return title.strip('.')

# ----------------------------------
# Processing Functions for Streamlit UI
# ----------------------------------

def process_single_video(url):
    """
    Downloads and processes a single YouTube video.
    Returns a tuple (pdf_data, pdf_filename) if successful,
    or (None, error_message) on failure.
    """
    try:
        # Download the video
        video_file = download_video(url, "video.mp4")
        video_title = get_video_title(url)
        output_pdf_name = f"{video_title}.pdf"
        with tempfile.TemporaryDirectory() as temp_folder:
            timestamps = extract_unique_frames(video_file, temp_folder)
            convert_frames_to_pdf(temp_folder, output_pdf_name, timestamps)
        # Read PDF content into memory
        with open(output_pdf_name, 'rb') as f:
            pdf_data = f.read()
        os.remove(video_file)
        os.remove(output_pdf_name)
        return pdf_data, output_pdf_name
    except Exception as e:
        return None, str(e)

def process_playlist(url):
    """
    Process a playlist URL by iterating over each video.
    Returns the zipped PDFs as an in-memory byte stream.
    """
    video_urls = get_playlist_videos(url)
    if not video_urls:
        raise Exception("No videos found in the playlist.")
    
    total_videos = len(video_urls)
    progress_bar = st.progress(0)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, video_url in enumerate(video_urls):
            st.write(f"Processing video: {video_url}")
            pdf_data, result = process_single_video(video_url)
            if pdf_data is not None:
                zipf.writestr(result, pdf_data)
            else:
                st.error(f"Failed to process {video_url}: {result}")
            progress_bar.progress((idx + 1) / total_videos)
    zip_buffer.seek(0)
    return zip_buffer

# ----------------------------------
# Streamlit UI
# ----------------------------------

# Optional custom CSS for a smoother look
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("YouTube Frame Extractor & PDF Generator")
st.write("""
This application downloads a YouTube video or playlist, extracts unique frames, and converts them into a PDF with embedded timestamps.
""")

url = st.text_input("Enter the YouTube Video or Playlist URL:")

if st.button("Process"):
    if not url:
        st.error("Please enter a URL.")
    else:
        with st.spinner("Processing, please wait..."):
            # Determine if the URL is for a single video or a playlist
            video_id = get_video_id(url)
            if video_id:
                pdf_data, result = process_single_video(url)
                if pdf_data is None:
                    st.error(f"Error processing video: {result}")
                else:
                    st.success("Video processed successfully!")
                    st.download_button("Download PDF", data=pdf_data, file_name=result, mime="application/pdf")
            else:
                try:
                    zip_data = process_playlist(url)
                    st.success("Playlist processed successfully!")
                    st.download_button("Download All PDFs (ZIP)", data=zip_data, file_name="videos.zip", mime="application/zip")
                except Exception as e:
                    st.error(f"Error processing playlist: {e}")
