from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
import uuid
import warnings
from datetime import datetime
from transformers import logging as hf_logging
import streamlit as st
from groq import Groq

# ---------------- STREAMLIT SETUP ----------------
st.set_page_config(page_title="Maharaj Impersonator", layout="wide")

# ---------------- SUPPRESS WARNINGS ----------------
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_TOKEN"] = "0"

import logging
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# ---------------- API CLIENT ----------------
api_key = st.secrets["API_KEY"]
client = Groq(api_key=api_key)

# ---------------- CHAT STORAGE ----------------
CHATS_DIR = "chats"
os.makedirs(CHATS_DIR, exist_ok=True)

def chat_path(chat_id):
    return os.path.join(CHATS_DIR, f"{chat_id}.json")

def save_chat(chat_id, title, messages):
    """Write a single chat thread to disk."""
    data = {
        "id": chat_id,
        "title": title,
        "updated_at": datetime.now().isoformat(),
        "messages": messages
    }
    with open(chat_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_chat(chat_id):
    """Load a single chat thread from disk. Returns None if not found."""
    path = chat_path(chat_id)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def delete_chat(chat_id):
    """Delete a chat file from disk."""
    path = chat_path(chat_id)
    if os.path.exists(path):
        os.remove(path)

def list_all_chats():
    """Return all chats sorted by most recently updated."""
    chats = []
    for fname in os.listdir(CHATS_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(CHATS_DIR, fname), encoding="utf-8") as f:
                    data = json.load(f)
                chats.append({
                    "id": data["id"],
                    "title": data.get("title", "Untitled"),
                    "updated_at": data.get("updated_at", "")
                })
            except Exception:
                pass
    chats.sort(key=lambda x: x["updated_at"], reverse=True)
    return chats

def make_title(first_message: str) -> str:
    """Auto-generate a thread title from the first user message."""
    return first_message.strip()[:50] + ("…" if len(first_message.strip()) > 50 else "")

def new_chat_id():
    return uuid.uuid4().hex[:12]

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

# ---------------- RAG + PROMPT FUNCTIONS ----------------
def get_similar(user_q, k=5):
    q_emb = model.encode([user_q])[0]
    scores = np.dot(embeddings, q_emb)
    top_k = np.argsort(scores)[-k:][::-1]
    return [pairs[i] for i in top_k]

def build_system_prompt(user_q, mode):
    examples = get_similar(user_q, 5)
    system = (
        "You are Maharaj — a deeply philosophical, reflective, and wise teacher. "
        "Answer every question in Maharaj's voice: thoughtful, spiritual, grounded in plenty of real-life examples. "
        "Below are example Q&A pairs to guide your tone and style:\n\n"
    )
    for ex in examples:
        system += f"Q: {ex['q']}\nA: {ex['a']}\n\n"

    if mode == "summary":
        system += (
            "\nIMPORTANT: Keep your answer concise — maximum 7 sentences. "
            "Be precise and do not exceed this limit."
        )
    else:
        system += "\nAnswer deeply and philosophically. Give plenty of real-life examples."

    return system

def get_answer(chat_history, user_q, mode):
    # Strip extra keys — Groq API only accepts "role" and "content"
    clean_history = [{"role": m["role"], "content": m["content"]} for m in chat_history[-4:]]
    system_prompt = build_system_prompt(user_q, mode)
    messages = [{"role": "system", "content": system_prompt}]
    messages += clean_history
    messages.append({"role": "user", "content": user_q})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content

# ---------------- SESSION STATE INIT ----------------
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "active_messages" not in st.session_state:
    st.session_state.active_messages = []
if "active_title" not in st.session_state:
    st.session_state.active_title = ""
if "pending_delete" not in st.session_state:
    st.session_state.pending_delete = None

def open_chat(chat_id):
    data = load_chat(chat_id)
    if data:
        st.session_state.active_chat_id = data["id"]
        st.session_state.active_messages = data["messages"]
        st.session_state.active_title = data["title"]

def start_new_chat():
    st.session_state.active_chat_id = new_chat_id()
    st.session_state.active_messages = []
    st.session_state.active_title = ""

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🕉️ Maharaj")
    st.markdown("---")

    if st.button("✏️ New Chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("---")
    st.markdown("**Previous Chats**")

    all_chats = list_all_chats()

    if not all_chats:
        st.caption("No saved chats yet.")
    else:
        for chat in all_chats:
            is_active = (chat["id"] == st.session_state.active_chat_id)
            col_title, col_del = st.sidebar.columns([5, 1])

            with col_title:
                # Bold label for active thread
                btn_label = f"**{chat['title']}**" if is_active else chat["title"]
                if st.button(btn_label, key=f"open_{chat['id']}", use_container_width=True):
                    open_chat(chat["id"])
                    st.rerun()

            with col_del:
                if st.button("🗑", key=f"del_{chat['id']}"):
                    st.session_state.pending_delete = chat["id"]
                    st.rerun()

    # Confirm delete dialog
    if st.session_state.pending_delete:
        del_id = st.session_state.pending_delete
        del_data = load_chat(del_id)
        del_title = del_data["title"] if del_data else del_id
        st.warning(f"Delete **{del_title}**?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, delete", use_container_width=True):
                delete_chat(del_id)
                st.session_state.pending_delete = None
                if st.session_state.active_chat_id == del_id:
                    st.session_state.active_chat_id = None
                    st.session_state.active_messages = []
                    st.session_state.active_title = ""
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.pending_delete = None
                st.rerun()

    st.markdown("---")
    st.caption("Chats are saved locally in the `chats/` folder.")

# ---------------- MAIN AREA ----------------

# No chat open — welcome screen
if st.session_state.active_chat_id is None:
    st.title("🕉️ Maharaj Impersonator")
    st.markdown("##### *A philosophical guide in the tradition of Nisargadatta Maharaj*")
    st.markdown("---")
    st.info("Click **✏️ New Chat** in the sidebar to begin, or select a previous conversation.")

else:
    # Active thread
    title_display = st.session_state.active_title or "New Conversation"
    st.title(f"🕉️ {title_display}")
    st.markdown("---")

    # ---------------- CHAT DISPLAY ----------------
    for msg in st.session_state.active_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🕉️"):
                if msg.get("mode") == "summary":
                    st.caption("📝 Summary mode")
                st.write(msg["content"])

    # ---------------- INPUT AREA ----------------
    st.markdown("---")
    user_q = st.text_area(
        "Your question:",
        height=100,
        placeholder="Ask Maharaj something...",
        key="user_input"
    )

    col1, col2, col3 = st.columns([3, 3, 2])
    with col1:
        submit_normal = st.button("✨ Ask Normally", use_container_width=True)
    with col2:
        submit_summary = st.button("📝 Ask as Summary", use_container_width=True)
    with col3:
        st.write("")  # spacer

    # ---------------- SUBMISSION HANDLER ----------------
    def handle_submit(mode):
        if not user_q.strip():
            st.warning("Please enter a question first.")
            return

        q = user_q.strip()

        # Set title from first message
        if not st.session_state.active_title:
            st.session_state.active_title = make_title(q)

        # Append user message
        st.session_state.active_messages.append({"role": "user", "content": q})

        # Get LLM answer — pass history minus the message we just appended
        with st.spinner("Maharaj is reflecting..."):
            prior = st.session_state.active_messages[:-1]
            answer = get_answer(prior, q, mode)

        # Append assistant message
        st.session_state.active_messages.append({
            "role": "assistant",
            "content": answer,
            "mode": mode
        })

        # Persist to disk immediately
        save_chat(
            st.session_state.active_chat_id,
            st.session_state.active_title,
            st.session_state.active_messages
        )

        st.rerun()

    if submit_normal and user_q.strip():
        handle_submit("normal")
    elif submit_summary and user_q.strip():
        handle_submit("summary")
    elif (submit_normal or submit_summary) and not user_q.strip():
        st.warning("Please enter a question first.")