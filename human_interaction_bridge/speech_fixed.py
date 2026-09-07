import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import numpy as np
import queue
import os
from collections import deque

DEVICE_ID = 1             # change this to match your mic (see sd.query_devices())
SILENCE_THRESHOLD = 100     # lower = more sensitive to quiet sound; raise if it stops too early
SEGMENT_SILENCE = 1.2       # seconds of silence that ends ONE subtitle segment
SESSION_SILENCE = 3.0       # seconds of silence (with no active segment) that ends the whole session
CHUNK_DURATION = 0.2        # how often (in seconds) we check the mic level
PRE_BUFFER_CHUNKS = 3       # chunks of audio kept BEFORE speech is detected
INITIAL_SPEECH_TIMEOUT = 15.0  # give up if NO speech at all happens within this long
TEMP_DIR = "segments_tmp"


def record_session():
    """
    Listens continuously and splits the audio into speech segments,
    each with an accurate (start_time, end_time) in seconds relative
    to the start of the session. This is what lets subtitles line up
    with the actual speech instead of being one big undifferentiated blob.
    """
    device_info = sd.query_devices(DEVICE_ID)
    sample_rate = int(device_info['default_samplerate'])
    print("Using device:", device_info['name'], "at", sample_rate, "Hz")

    chunk_samples = int(CHUNK_DURATION * sample_rate)
    segment_silence_chunks = int(SEGMENT_SILENCE / CHUNK_DURATION)
    session_silence_chunks = int(SESSION_SILENCE / CHUNK_DURATION)

    sd.rec(int(0.3 * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=DEVICE_ID)
    sd.wait()

    print("Listening... start speaking whenever you're ready. (long pause = stop)")

    audio_queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        audio_queue.put(indata.copy())

    pre_buffer = deque(maxlen=PRE_BUFFER_CHUNKS)
    segments = []  # list of (start_time, end_time, np.ndarray)

    started_speaking = False
    segment_silent_count = 0
    session_silent_count = 0
    segment_start_time = 0.0
    current_segment_chunks = []
    chunk_index = 0

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype='int16',
        device=DEVICE_ID,
        blocksize=chunk_samples,
        callback=callback,
    )

    with stream:
        while True:
            try:
                chunk = audio_queue.get(timeout=1.0)
            except queue.Empty:
                chunk = None

            timestamp = chunk_index * CHUNK_DURATION

            if not started_speaking and not segments and timestamp >= INITIAL_SPEECH_TIMEOUT:
                print(f"No speech detected within {INITIAL_SPEECH_TIMEOUT:.0f}s, giving up.")
                break

            if chunk is None:
                continue

            chunk_index += 1
            volume = np.abs(chunk).mean()
            print(volume)
            if volume > SILENCE_THRESHOLD:
                if not started_speaking:
                    segment_start_time = max(0.0, timestamp - PRE_BUFFER_CHUNKS * CHUNK_DURATION)
                    current_segment_chunks = list(pre_buffer)
                started_speaking = True
                segment_silent_count = 0
                session_silent_count = 0
                current_segment_chunks.append(chunk)
            elif started_speaking:
                segment_silent_count += 1
                current_segment_chunks.append(chunk)
                if segment_silent_count >= segment_silence_chunks:
                    segment_end_time = timestamp
                    segment_audio = np.concatenate(current_segment_chunks)
                    segments.append((segment_start_time, segment_end_time, segment_audio))
                    started_speaking = False
                    current_segment_chunks = []
            else:
                pre_buffer.append(chunk)
                session_silent_count += 1
                if segments and session_silent_count >= session_silence_chunks:
                    break

    return segments, sample_rate


def clean_stutter_transcript(text):
    if not text:
        return text
    words = text.split()
    cleaned = []
    for word in words:
        if cleaned and cleaned[-1].lower() == word.lower():
            continue
        cleaned.append(word)
    return " ".join(cleaned)


def transcribe_segment(recording, sample_rate, path):
    wav.write(path, sample_rate, recording)

    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        # Helps the recognizer adjust to background noise before reading
        # the actual speech, which can improve accuracy.
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio = recognizer.record(source)

    try:
        # language="en-IN" tells Google to expect Indian English pronunciation
        # instead of defaulting to American English — this alone often fixes
        # a lot of "inaudible" results for Indian English speakers.
        return recognizer.recognize_google(audio, language="en-IN")
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print("Speech recognition service error:", e)
        return None


def format_srt_timestamp(seconds):
    ms_total = int(round(seconds * 1000))
    hh, ms_total = divmod(ms_total, 3600000)
    mm, ms_total = divmod(ms_total, 60000)
    ss, ms = divmod(ms_total, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def write_srt(subtitle_lines, filename="subtitles.srt"):
    with open(filename, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(subtitle_lines, 1):
            f.write(f"{i}\n")
            f.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            f.write(f"{text}\n\n")


if __name__ == "__main__":
    os.makedirs(TEMP_DIR, exist_ok=True)

    segments, sample_rate = record_session()
    print(f"Done listening. Captured {len(segments)} speech segment(s). Transcribing...")

    if not segments:
        print("No speech detected.")
    else:
        subtitle_lines = []
        for i, (start, end, recording) in enumerate(segments):
            seg_path = os.path.join(TEMP_DIR, f"segment_{i}.wav")
            raw_text = transcribe_segment(recording, sample_rate, seg_path)

            if raw_text:
                text = clean_stutter_transcript(raw_text)
            else:
                text = "[inaudible]"

            print(f"[{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}] {text}")
            subtitle_lines.append((start, end, text))

        write_srt(subtitle_lines, "subtitles.srt")
        print("\nSaved subtitles.srt")
