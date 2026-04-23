from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
import warnings
from transformers import logging as hf_logging
import streamlit as st

from groq import Groq

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="Maharaj Impersonator", layout="centered")

st.title("Maharaj Impersonator")

api_key = st.secrets["API_KEY"]
client = Groq(api_key=api_key)

# ---------------- SUPPRESS WARNINGS ----------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_TOKEN"] = "0"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    with open("dataset.json", encoding="utf-8") as f:
        return json.load(f)

pairs = load_data()
questions = [p["q"] for p in pairs]

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------- LOAD / CREATE EMBEDDINGS ----------------
@st.cache_data
def load_embeddings():
    if os.path.exists("embeddings.npy"):
        return np.load("embeddings.npy")
    else:
        emb = model.encode(questions)
        np.save("embeddings.npy", emb)
        return emb

embeddings = load_embeddings()

# ---------------- CORE FUNCTIONS (UNCHANGED LOGIC) ----------------
def get_similar(user_q, k=5):
    q_emb = model.encode([user_q])[0]
    scores = np.dot(embeddings, q_emb)
    top_k = np.argsort(scores)[-k:][::-1]
    return [pairs[i] for i in top_k]

def build_prompt(user_q, mode="normal"):
    examples = get_similar(user_q, 5)

    if mode == "summary":
        instruction = (
            "Answer in the same philosophical style.\n"
            "IMPORTANT: Limit your answer to MAXIMUM 7 sentences.\n"
            "Do NOT exceed 7 sentences."
        )
    else:
        instruction = "Answer deeply, philosophically, and in the same reflective tone. IMPORTANT: Give plenty of real life examples."

    prompt = instruction + "\n\n"

    for ex in examples:
        prompt += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

    prompt += f"Q: {user_q}\nA:"
    return prompt

# ---------------- SESSION STATE ----------------
if "mode" not in st.session_state:
    st.session_state.mode = "normal"

# ---------------- UI ----------------


user_q = st.text_area("Ask your question:", height=120)




# ---------------- PROCESS INPUT ----------------
col1, col2 = st.columns([1,7])
with col2:
    if st.button("Switch Mode"):
        st.session_state.mode = "summary" if st.session_state.mode == "normal" else "normal"
       

st.write(f"**Current Mode:** {st.session_state.mode}")

submit_clicked = False

with col1:
    if st.button("Submit") and user_q:
        submit_clicked = True

if submit_clicked:
    prompt = build_prompt(user_q, st.session_state.mode)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    st.subheader("Answer")

    if st.session_state.mode == 'summary':
        original_text = response.choices[0].message.content

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that summarizes text concisely while keeping key points."
                },
                {
                    "role": "user",
                    "content": f"Please summarize the following text:\n\n{original_text}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            max_tokens=500
        )

        summary = chat_completion.choices[0].message.content
        st.write(summary)
    else:
        st.write(response.choices[0].message.content)