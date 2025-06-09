import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
from gtts import gTTS
from playsound import playsound
import os

# Load model
MODEL_PATH = "cnn_audio_model.h5"
model = load_model(MODEL_PATH)

# Judul halaman
st.set_page_config(page_title="Audio Classifier", layout="centered")
st.title("🎵 Prediksi Suara Lingkungan Sekitar")
st.write("Upload file audio (.wav) dan sistem akan memprediksi jenis suaranya.")

# Upload audio file
uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

# Label mapping dari classID (dengan terjemahan ke Bahasa Indonesia)
label_map = {
    0: "AC (pendingin ruangan)",
    1: "klakson mobil",
    2: "anak-anak bermain",
    3: "gonggongan anjing",
    4: "mesin bor",
    5: "mesin menyala diam",
    6: "tembakan",
    7: "bor jalan (jackhammer)",
    8: "sirine",
    9: "musik jalanan"
}

# Fungsi ekstraksi fitur MFCC
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Terjadi error saat membaca file: {e}")
        return None

# Fungsi text-to-speech (gTTS Bahasa Indonesia)
def speak(text, lang="id"):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        playsound("output.mp3")
        os.remove("output.mp3")  # hapus file setelah diputar
    except Exception as e:
        st.error(f"Text-to-Speech gagal: {e}")

# Prediksi saat file di-upload
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Memproses audio...")

    # Simpan file temporer
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Ekstrak fitur
    mfcc = extract_features("temp.wav")
    if mfcc is not None:
        mfcc = mfcc.reshape(1, 40, 1)
        prediction = model.predict(mfcc)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        hasil_prediksi = label_map[predicted_class]
        st.success(f"Prediksi: {hasil_prediksi}")
        st.write(f"Tingkat keyakinan: {confidence:.2f}")

        # Text-to-speech dalam Bahasa Indonesia
        kalimat = f"Suara yang diprediksi adalah {hasil_prediksi} dengan keyakinan {confidence:.0%}"
        speak(kalimat)
