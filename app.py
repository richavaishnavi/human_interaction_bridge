import streamlit as st
import time
from brain import clean_and_speak

st.set_page_config(page_title="Communication Bridge AI", page_icon="🌉", layout="centered")

# Custom font import
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗨️ Unspoken")
st.markdown("### Convert sign language, broken speech, gestures or simple expressions to fluent speech")
st.divider()
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(circle, #312e81 1px, transparent 1px);
        background-size: 24px 24px;
    }
    </style>
""", unsafe_allow_html=True)

# Mode toggle
mode = st.toggle("🖐️ Gesture mode (off = Speech mode)")
source_label = "gesture" if mode else "speech"

if mode:
    st.caption("🖐️ Gesture mode — Person 2's webcam detection will feed words here.")
    sample_inputs = {
        "👋 Wave": "hello",
        "👍 Thumbs up": "yes",
        "✋ Open palm": "stop",
    }
else:
    st.caption("🎙️ Speech mode — try a sample of messy, disfluent speech.")
    sample_inputs = {
        "🗣️ Stuttering speech": "um i i want want to go go to the store",
        "❓ Broken sentence": "where where is is the bathroom",
        "🙏 Simple request": "help me please",
    }

choice = st.selectbox("Pick a sample input:", list(sample_inputs.keys()))
raw_text = sample_inputs[choice]

if st.button("✨ Convert to fluent speech", use_container_width=True):
    progress = st.progress(0, text="Warming up...")
    for pct, msg in [(30, "Sending to AI..."), (70, "Polishing sentence..."), (100, "Done!")]:
        time.sleep(0.3)
        progress.progress(pct, text=msg)

    result = clean_and_speak({"source": source_label, "raw_text": raw_text})
    progress.empty()

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown("#### 📝 Raw Input")
        st.container(border=True).write(raw_text)
    with col2:
        st.markdown("#### ✨ Fluent Output")
        st.container(border=True).write(result["clean_text"])

    word_diff = len(raw_text.split()) - len(result["clean_text"].split())
    st.metric(label="Words simplified", value=len(result["clean_text"].split()), delta=f"{-word_diff} vs raw")

    st.balloons()
    st.markdown("🔊 **Spoken aloud!**")
else:
    st.info(f"Raw input ready: *{raw_text}*")