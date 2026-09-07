"""
Speech Input Module
--------------------
Listens to the microphone and transcribes a spoken sentence to text.

Two modes:
  1. "sr"      -> Uses the SpeechRecognition library (default).
                  - Works offline with the google recognizer swapped for
                    recognize_sphinx() if you install pocketsphinx.
                  - By default uses Google's free web API (needs internet,
                    no API key required, rate-limited).
  2. "whisper" -> Uses OpenAI's Whisper API (needs OPENAI_API_KEY env var).
                  Slower to set up but noticeably better accuracy, especially
                  with accents, background noise, or domain-specific words.

Usage:
    python speech_input.py            # SpeechRecognition mode (default)
    python speech_input.py --mode whisper

Install (choose based on mode):
    pip install SpeechRecognition pyaudio
    # offline recognizer (optional):
    pip install pocketsphinx
    # whisper mode:
    pip install openai
"""

import argparse
import sys
import tempfile
import os
import re
from datetime import datetime


def listen_and_transcribe_sr(offline: bool = False, timeout: int = 5) -> str:
    """
    Capture one spoken sentence from the default microphone and return the
    transcribed text, using the SpeechRecognition library.
    """
    import speech_recognition as sr

    recognizer = sr.Recognizer()
    # Allow longer pauses before assuming the sentence is finished, and
    # keep adapting to background noise instead of locking in one reading.
    recognizer.pause_threshold = 1.2
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("Adjusting for ambient noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... speak your sentence now.")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=None)
        except sr.WaitTimeoutError:
            return "[ERROR] No speech detected within timeout."

    print("Transcribing...")
    try:
        if offline:
            # Requires: pip install pocketsphinx
            text = recognizer.recognize_sphinx(audio)
        else:
            # Free Google web API, requires internet, no key needed
            text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "[ERROR] Could not understand audio."
    except sr.RequestError as e:
        return f"[ERROR] API request failed: {e}"


def listen_and_transcribe_whisper(timeout: int = 5) -> str:
    """
    Capture one spoken sentence from the default microphone and transcribe it
    using OpenAI's Whisper API. Requires OPENAI_API_KEY to be set.
    """
    import speech_recognition as sr
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "[ERROR] OPENAI_API_KEY environment variable not set."

    client = OpenAI(api_key=api_key)
    recognizer = sr.Recognizer()
    # Allow longer pauses before assuming the sentence is finished, and
    # keep adapting to background noise instead of locking in one reading.
    recognizer.pause_threshold = 1.2
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("Adjusting for ambient noise... please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening... speak your sentence now.")
        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=None)
        except sr.WaitTimeoutError:
            return "[ERROR] No speech detected within timeout."

    # Whisper API needs an audio file, so write the captured audio to a temp WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio.get_wav_data())
        tmp_path = tmp.name

    print("Transcribing via Whisper API...")
    try:
        with open(tmp_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )
        return transcript.text
    finally:
        os.remove(tmp_path)


FILLER_WORDS = [
    "um", "uh", "uhh", "umm", "erm", "er", "ah",
    "like", "you know", "i mean", "sort of", "kind of",
    "basically", "actually", "literally", "so yeah",
]


def clean_transcript(text: str) -> str:
    """
    Take a raw transcript and return a cleaner version:
      - removes stutter/word repetitions ("I I I want" -> "I want")
      - removes common filler words/phrases ("um", "like", "you know", etc.)
      - collapses extra whitespace and fixes spacing around punctuation
    This is a rule-based pass (no API calls), so it's fast and offline.
    """
    if not text or text.startswith("[ERROR]"):
        return text

    cleaned = text

    # 1. Remove immediate word-level stutters/repeats, e.g. "I I I want" -> "I want"
    cleaned = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', cleaned, flags=re.IGNORECASE)

    # 2. Remove filler words/phrases (case-insensitive, whole-word/phrase match)
    for filler in sorted(FILLER_WORDS, key=len, reverse=True):
        pattern = r'(?<!\w)' + re.escape(filler) + r'(?!\w)'
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # 3. Collapse leftover extra spaces/commas created by removals
    cleaned = re.sub(r'\s+,', ',', cleaned)
    cleaned = re.sub(r',\s*,', ',', cleaned)
    cleaned = re.sub(r'\s{2,}', ' ', cleaned)
    cleaned = cleaned.strip(" ,")

    # 4. Capitalize first letter
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned


def save_transcript(raw: str, cleaned: str, out_path: str = "transcript.txt") -> str:
    """
    Append the raw and cleaned transcript (with a timestamp) to a text file.
    Returns the path written to.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(f"--- {timestamp} ---\n")
        f.write(f"Raw:     {raw}\n")
        f.write(f"Cleaned: {cleaned}\n\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Mic input -> text transcription")
    parser.add_argument(
        "--mode",
        choices=["sr", "whisper"],
        default="sr",
        help="Transcription backend: 'sr' (SpeechRecognition, default) or 'whisper' (OpenAI Whisper API)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use offline recognizer (pocketsphinx) instead of Google web API. Only applies to --mode sr.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Seconds to wait for speech to start before giving up (default: 5)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="transcript.txt",
        help="File to append raw + cleaned transcripts to (default: transcript.txt)",
    )
    args = parser.parse_args()

    if args.mode == "sr":
        raw = listen_and_transcribe_sr(offline=args.offline, timeout=args.timeout)
    else:
        raw = listen_and_transcribe_whisper(timeout=args.timeout)

    cleaned = clean_transcript(raw)

    print("\n--- Raw Transcript ---")
    print(raw)
    print("\n--- Cleaned Transcript ---")
    print(cleaned)

    if not raw.startswith("[ERROR]"):
        path = save_transcript(raw, cleaned, args.out)
        print(f"\nSaved to {path}")

    return cleaned


if __name__ == "__main__":
    main()
