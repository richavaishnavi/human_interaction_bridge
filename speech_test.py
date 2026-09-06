import speech_recognition as sr
from brain import clean_and_speak, speak

def listen_and_transcribe():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening... speak now")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=5, phrase_time_limit=10)
    try:
        text = r.recognize_google(audio)
        print("You said:", text)
    except sr.UnknownValueError:
        text = ""
        print("Could not understand audio")
    except sr.RequestError as e:
        text = ""
        print(f"Speech recognition error: {e}")
    return {"source": "speech", "raw_text": text}

if __name__ == "__main__":
    input_data = listen_and_transcribe()
    if input_data["raw_text"]:
        result = clean_and_speak(input_data)
        print("Cleaned:", result["clean_text"])
        speak(result["clean_text"])
    else:
        print("Nothing to process.")