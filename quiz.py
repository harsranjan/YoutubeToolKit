import streamlit as st
from transformers import pipeline, T5Tokenizer
import spacy
import nltk
import torch

# Download required nltk tokenizer data
nltk.download('punkt')

# Load the spaCy language model
nlp = spacy.load("en_core_web_sm")

# Load the T5 tokenizer
tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-small-qg-hl", use_fast=False)

# Initialize the question-generation pipeline.
# The parameters `device_map="auto"` and `torch_dtype=torch.float32` ensure proper loading,
# avoiding the casting issue with GPTQ models.
qg_pipeline = pipeline(
    "text2text-generation",
    model="valhalla/t5-small-qg-hl",
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float32
)

# Initialize the summarization pipeline similarly.
summarizer_pipeline = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device_map="auto",
    torch_dtype=torch.float32,
    use_fast=True
)

st.set_page_config(page_title="YT Quiz & Notes Generator", layout="centered")

st.markdown(
    """
    <style>
    .flashcard {
        background-color: #e6f0ff;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .flashcard h4 {
        margin-bottom: 5px;
    }
    .note-box {
        background-color: #fff5e6;
        padding: 20px;
        border-left: 5px solid #ff9900;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def generate_flashcards(summary_text, min_words=5, max_questions=50):
    flashcards = []
    doc = nlp(summary_text)
    
    for sent in doc.sents:
        if len(flashcards) >= max_questions:
            break

        sent_text = sent.text.strip()
        if len(sent_text.split()) < min_words:
            continue

        # Use a named entity if available for more relevant questions; otherwise fall back on noun chunks.
        if sent.ents:
            candidate = sent.ents[0].text.strip()
        else:
            noun_chunks = list(sent.noun_chunks)
            candidate = noun_chunks[0].text.strip() if noun_chunks else sent_text.split()[0]

        # Highlight the candidate in the sentence.
        if candidate in sent_text:
            highlighted_sent = sent_text.replace(candidate, f"<hl> {candidate} <hl>", 1)
        else:
            highlighted_sent = sent_text

        input_text = f"generate question: {highlighted_sent}"
        try:
            output = qg_pipeline(input_text)
            generated = output[0]['generated_text'].strip()
            # If the generated output follows a Q:/A: format, extract question and answer.
            if "Q:" in generated and "A:" in generated:
                q_part = generated.split("A:")[0].replace("Q:", "").strip()
                a_part = generated.split("A:")[1].strip()
                if q_part and a_part:
                    flashcards.append({"question": q_part, "answer": sent_text})
                    continue
            if generated:
                flashcards.append({"question": generated, "answer": sent_text})
        except Exception as e:
            st.error(f"Error generating flashcard for sentence:\n{sent_text}\nError: {e}")

    return flashcards[:max_questions]

def generate_study_notes(summary_text):
    try:
        summary_output = summarizer_pipeline(
            summary_text,
            max_length=200,
            min_length=50,
            do_sample=False,
            truncation=True
        )
        if summary_output and len(summary_output) > 0 and 'summary_text' in summary_output[0]:
            return summary_output[0]['summary_text']
        else:
            return "No study notes could be generated from the summary."
    except Exception as e:
        return f"Error generating study notes: {e}"

def main():
    st.title("\U0001F4D6 YouTube Quiz & Notes Generator")
    st.markdown(
        """
        Enhance your learning with auto-generated flashcards and summarized notes from your YouTube transcript!

        **Instructions:**
        1. Paste the transcript summary below.
        2. Click **Generate Quiz and Study Notes**.
        3. Review the questions and concise notes.
        """
    )

    st.markdown("---")
    summary_text = st.text_area(
        "Paste Transcript Summary Here:",
        height=200,
        help="Enter the transcript summary from your YouTube video."
    )

    col1, col2, _ = st.columns([1,1,2])
    with col1:
        generate_btn = st.button("\U0001F4DD Generate Quiz and Study Notes")

    if generate_btn:
        if not summary_text.strip():
            st.error("Please provide a transcript summary.")
            return

        with st.spinner("Generating flashcards..."):
            flashcards = generate_flashcards(summary_text)
        st.success("Flashcards ready!")

        with st.spinner("Creating summarized study notes..."):
            study_notes = generate_study_notes(summary_text)
        st.success("Study notes generated!")

        st.markdown("---")
        st.header("\U0001F4DA Flashcards")
        if flashcards:
            for idx, card in enumerate(flashcards, start=1):
                st.markdown(f"<div class='flashcard'>", unsafe_allow_html=True)
                st.subheader(f"Flashcard {idx}")
                st.markdown("<h4>Question:</h4>", unsafe_allow_html=True)
                st.write(card["question"])
                with st.expander("Show Answer"):
                    st.markdown("<h4>Answer:</h4>", unsafe_allow_html=True)
                    st.write(card["answer"])
                st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No flashcards were generated. Try refining your transcript.")

        st.markdown("---")
        st.header("\U0001F4CB Study Notes")
        st.markdown(f"<div class='note-box'>{study_notes}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
