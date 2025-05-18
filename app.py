import requests
import streamlit as st

# Use token from Streamlit Cloud Secrets (recommended)
hf_token = st.secrets["HUGGINGFACE_TOKEN"]

# Define the model and API endpoint
model_id = "facebook/bart-large-cnn"
API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

headers = {
    "Authorization": f"Bearer {hf_token}"
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

# Streamlit UI
st.set_page_config(page_title="Medical Note Summarizer", layout="centered")
st.title("🩺 Medical Note Summarizer")
st.write("Enter a medical note below and get a concise summary using a Hugging Face model.")

note_input = st.text_area("Medical Note", height=250, placeholder="Paste your medical note here...")

if st.button("Summarize"):
    if note_input.strip() == "":
        st.warning("Please enter a medical note before summarizing.")
    else:
        with st.spinner("Summarizing..."):
            try:
                result = query({"inputs": note_input})
                summary = result[0]['summary_text']
                st.success("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
