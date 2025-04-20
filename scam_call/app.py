import streamlit as st
import pickle
import joblib
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
import os

# Load SVM model and TF-IDF vectorizer
with open("svm_model.pkl", "rb") as f:
    model = pickle.load(f)

vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Prediction function
def predict(text):
    transformed = vectorizer.transform([text])
    return model.predict(transformed)[0]

# Transcription helper
def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Could not understand audio"

# Streamlit UI
st.title("🕵️ Fraud/Scam Detection App")

option = st.sidebar.radio("Choose Input Type", ["Type Text", "Upload Text File", "Real-Time Audio"])

# Option 1: Manual text input
if option == "Type Text":
    user_input = st.text_area("Enter text:")
    if st.button("Analyze"):
        result = predict(user_input)
        st.success(f"Prediction: {'🚨 Scam/Fraud' if result == 1 else '✅ Not Fraud'}")

# Option 2: Upload .txt file
elif option == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        st.text_area("File Content", content, height=200)
        if st.button("Analyze Text File"):
            result = predict(content)
            st.success(f"Prediction: {'🚨 Scam/Fraud' if result == 1 else '✅ Not Fraud'}")

# Option 3: Real-time audio
elif option == "Real-Time Audio":
    st.info("Click to record.")
    if st.button("Start Recording"):
        import sounddevice as sd
        import soundfile as sf
        fs = 44100
        duration = 8
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        tmp_wav = "temp_record.wav"
        sf.write(tmp_wav, audio_data, fs)
        st.audio(tmp_wav)
        text = transcribe_audio(tmp_wav)
        st.write(f"Transcribed Text: `{text}`")
        result = predict(text)
        st.success(f"Prediction: {'🚨 Scam/Fraud' if result == 1 else '✅ Not Fraud'}")
        os.remove(tmp_wav)
