import streamlit as st
import time
import speech_recognition as sr
from brain import clean_and_speak
from gestures_emotions import get_current_input

st.set_page_config(page_title="Unspoken", page_icon="🗨️", layout="centered")

# Custom font import + white text so everything is readable on the dark background
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .stApp * {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🗨️ Unspoken")
st.markdown("### Convert sign language, gestures, broken speech or simply emotions to fluent speech and text")
st.markdown("""
    <style>
    .stApp {
        background-color: #0a0e1a;
        background-image:
            radial-gradient(circle at 0% 100%, rgba(34, 197, 94, 0.35) 0%, transparent 28%),
            radial-gradient(circle at 35% 75%, rgba(236, 72, 153, 0.35) 0%, transparent 28%),
            radial-gradient(circle at 70% 40%, rgba(59, 130, 246, 0.30) 0%, transparent 30%);
        background-size: 200% 200%;
        background-position: 0% 100%;
        animation: auroraDrift 5s ease-in-out infinite alternate;
    }
    @keyframes auroraDrift {
        0% { background-position: 0% 100%; }
        100% { background-position: 100% 0%; }
    }
    </style>
""", unsafe_allow_html=True)
st.divider()


def listen_and_transcribe():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    try:
        return r.recognize_google(audio)
    except Exception:
        return ""


# Mode toggle
mode = st.toggle("🖐️ Gesture mode (off = Speech mode)")
source_label = "gesture" if mode else "speech"

raw_text = None

if mode:
    st.caption("🖐️ Gesture mode — show a gesture and expression to your webcam")

    if st.button("📷 Capture gesture + emotion", use_container_width=True):
        with st.spinner("Opening webcam and detecting..."):
            detected = get_current_input()

        gesture = detected.get("gesture")
        emotion = detected.get("emotion", "Neutral")

        if gesture:
            raw_text = gesture
            st.success(f"Detected gesture: **{gesture}** | Emotion: **{emotion}**")
        else:
            raw_text = emotion  # fall back to emotion alone if no clear gesture
            st.info(f"No clear gesture detected. Emotion: **{emotion}**")
else:
    st.caption("🎙️ Speech mode — press the button and speak clearly")

    mic_placeholder = st.empty()
    mic_placeholder.markdown(
        "<div style='text-align:center; font-size:5rem;'>🎤</div>",
        unsafe_allow_html=True
    )

    if st.button("🔴 Start Recording", use_container_width=True):
        mic_placeholder.markdown(
            "<div style='text-align:center; font-size:5rem; color:#22c55e;'>🎤</div>",
            unsafe_allow_html=True
        )
        st.caption("Listening... speak now")
        raw_text = listen_and_transcribe()
        mic_placeholder.markdown(
            "<div style='text-align:center; font-size:5rem;'>🎤</div>",
            unsafe_allow_html=True
        )
        if raw_text:
            st.success(f"Heard: {raw_text}")
        else:
            st.warning("Didn't catch that, try again.")

if raw_text and st.button("✨ Convert to fluent speech", use_container_width=True):
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
elif raw_text:
    st.info(f"Raw input ready: *{raw_text}*")