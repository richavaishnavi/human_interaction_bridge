import google.generativeai as genai
import pyttsx3
import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-3.6-flash")

def clean_and_speak(input_dict):
    raw = input_dict.get("raw_text", "")
    if not raw.strip():
        return {"clean_text": ""}
    
    try:
        response = model.generate_content(
            f"Rewrite this into one clear, fluent, natural sentence. Only output the sentence, nothing else:\n\n{raw}"
        )
        clean_text = response.text.strip()
    except Exception as e:
        print(f"API failed, using fallback: {e}")
        clean_text = raw.strip().capitalize()
        if clean_text and not clean_text.endswith((".", "!", "?")):
            clean_text += "."
    
    return {"clean_text": clean_text}

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    test_input = {"source": "speech", "raw_text": "um i i want want to go go to store"}
    result = clean_and_speak(test_input)
    print(result)
    speak(result["clean_text"])