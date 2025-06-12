import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
from gtts import gTTS
from playsound import playsound
import os

MODEL_PATH = "cnn_audio_model.h5"
model = load_model(MODEL_PATH)

st.set_page_config(page_title="Audio Classifier", layout="centered")
st.title("🔊 Deteksi Suara Lingkungan Berbahaya")
st.write("Upload file audio (.wav) dan sistem akan mendeteksi apakah suara tersebut **berbahaya** atau **tidak berbahaya**.")

uploaded_file = st.file_uploader("Upload audio file", type=["wav"])

label_map = {
    0: "AC (pendingin ruangan)",
    1: "klakson mobil",
    2: "anak-anak bermain",
    3: "gonggongan anjing",
    4: "mesin bor",
    5: "suara kendaraan",
    6: "tembakan",
    7: "bor jalan (jackhammer)",
    8: "sirine",
    9: "musik jalanan"
}

# Daftar suara yang dianggap berbahaya
dangerous_sounds = {
    "klakson mobil",
    "gonggongan anjing",
    "mesin bor",
    "tembakan",
    "bor jalan (jackhammer)",
    "sirine"
}

def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
        return mfccs_scaled
    except Exception as e:
        st.error(f"Terjadi error saat membaca file: {e}")
        return None

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    st.write("Memproses audio...")

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    mfcc = extract_features("temp.wav")
    if mfcc is not None:
        mfcc = mfcc.reshape(1, 40, 1)
        prediction = model.predict(mfcc)
        predicted_class = np.argmax(prediction)

        hasil_prediksi = label_map[predicted_class]
        status_bahaya = "Berbahaya" if hasil_prediksi in dangerous_sounds else "Tidak Berbahaya"

        if status_bahaya == "Berbahaya":
            st.error(f"🚨 Prediksi: {hasil_prediksi} → {status_bahaya}")
        else:
            st.success(f"✅ Prediksi: {hasil_prediksi} → {status_bahaya}")

