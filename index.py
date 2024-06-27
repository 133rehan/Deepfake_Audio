from io import BytesIO
import requests
import streamlit as st
import numpy as np
import os
import joblib
import librosa
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Loading model
models_path = r"D:\FYP\DFAS\models\fixed\f"
xgboost_model = joblib.load(os.path.join(models_path, 'FXG Boost.pkl'))
gradient_boosting_model = joblib.load(os.path.join(models_path, 'FGradient_Boosting.pkl'))
random_forest_model = joblib.load(os.path.join(models_path, 'FRandom_Forest.pkl'))

# Assuming highly_correlated_indices contains the indices of retained features
highly_correlated_indices = [155, 156, 158, 157, 1, 3, 16, 17, 15, 21, 22, 18, 14, 20, 19, 13, 23, 24]

# Function to zero pad or truncate the signal
def zero_pad(signal, target_length):
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
    return signal

def truncate(signal, target_length):
    if len(signal) > target_length:
        signal = signal[:target_length]
    return signal

# Function to extract features from the audio file
def extract_features(file_path, fixed_length=31, target_sr=16000):
    y, sr = librosa.load(file_path, sr=None)
    
    # Resample to the target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Calculate the target length in samples
    target_length = fixed_length * sr

    # Ensure zero-padding or truncation
    y = zero_pad(truncate(y, target_length), target_length)

    # Determine appropriate n_fft
    n_fft = min(512, len(y))

    # Compute the STFT and magnitude spectrogram
    stft = librosa.stft(y, n_fft=n_fft)
    magnitude = np.abs(stft)

    # Features to extract
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    chroma = np.mean(librosa.feature.chroma_stft(S=magnitude, sr=sr, n_fft=min(n_fft, len(y))), axis=1)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=min(n_fft, len(y))), axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(S=magnitude, sr=sr, n_fft=min(n_fft, len(y))), axis=1)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr), axis=1)

    features = np.hstack((mfccs, chroma, mel, contrast, tonnetz))
    selected_features = features[highly_correlated_indices]

    return selected_features, y, sr

# Function to extract audio from video
def extract_audio_from_video(video_file):
    video = VideoFileClip(video_file)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# Getting confidence score from model
def predict_audio(audio_file, model):
    features, y, sr = extract_features(audio_file)
    features = features.reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)
    else:
        prediction = model.predict(features)[0]
        confidence = None  # If no probabilities, confidence remains undefined
    return prediction, confidence, y, sr

# UI
st.set_page_config(layout="wide", page_title="Deepfake Audio Detection", page_icon="")

# Sidebar content
audio_source = st.sidebar.selectbox("Select audio source:", ("Local Upload", "Internet Link", "Video File"))
audio_file = None
temp_files = []  # List to keep track of temporary files

if audio_source == "Local Upload":
    audio_file = st.sidebar.file_uploader("Upload your audio file", type=None)
    if audio_file is not None:
        # Workaround to keep file accessible
        audio_bytes = audio_file.read()
        audio_path = "uploaded_audio.mp3"  # Temporary file path
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        st.sidebar.audio(BytesIO(audio_bytes), format="audio/mp3")
        temp_files.append(audio_path)

elif audio_source == "Internet Link":
    audio_link = st.sidebar.text_input("Enter the URL of the audio file")
    if audio_link:
        try:
            audio_data = requests.get(audio_link).content
            audio_file = BytesIO(audio_data)
            audio_path = "downloaded_audio.mp3"  # Temporary file path
            with open(audio_path, 'wb') as f:
                f.write(audio_data)
            st.sidebar.audio(audio_file)
            temp_files.append(audio_path)
        except requests.RequestException as e:
            st.sidebar.error(f"Failed to download audio: {str(e)}")

elif audio_source == "Video File":
    video_file = st.sidebar.file_uploader("Upload your video file", type=['mp4', 'avi', 'mov'])
    if video_file is not None:
        # Display message while extracting audio
        with st.spinner("Extracting audio..."):
            # Extract audio from video
            video_path = video_file.name
            with open(video_path, 'wb') as f:
                f.write(video_file.read())
            audio_path = extract_audio_from_video(video_path)
            st.sidebar.audio(audio_path, format='audio/wav')
            temp_files.append(video_path)
            temp_files.append(audio_path)

# Main content
st.title("Deepfake Audio Detection")

if st.sidebar.button("Analyze"):
    if audio_path:
        with st.spinner("Analyzing audio..."):
            xgboost_prediction, xgboost_confidence, y, sr = predict_audio(audio_path, xgboost_model)
            gradient_boosting_prediction, gradient_boosting_confidence, _, _ = predict_audio(audio_path, gradient_boosting_model)
            random_forest_prediction, random_forest_confidence, _, _ = predict_audio(audio_path, random_forest_model)

            # Display the audio analysis result for XGBoost
            if xgboost_prediction == 1:
                st.error(f"XGBoost Model: Your audio is Fake with Confidence of {xgboost_confidence*100:.2f}%")
            elif xgboost_prediction == 0:
                st.success(f"XGBoost Model: Your audio is Real with Confidence of {xgboost_confidence*100:.2f}%")
            else:
                st.warning("XGBoost Model: The analysis is inconclusive.")
            
            # Display the audio analysis result for Gradient Boosting
            if gradient_boosting_prediction == 1:
                st.error(f"Gradient Boosting Model: Your audio is Fake with Confidence of {gradient_boosting_confidence*100:.2f}%")
            elif gradient_boosting_prediction == 0:
                st.success(f"Gradient Boosting Model: Your audio is Real with Confidence of {gradient_boosting_confidence*100:.2f}%")
            else:
                st.warning("Gradient Boosting Model: The analysis is inconclusive.")
            
            # Display the audio analysis result for Random Forest
            if random_forest_prediction == 1:
                st.error(f"Random Forest Model: Your audio is Fake with Confidence of {random_forest_confidence*100:.2f}%")
            elif random_forest_prediction == 0:
                st.success(f"Random Forest Model: Your audio is Real with Confidence of {random_forest_confidence*100:.2f}%")
            else:
                st.warning("Random Forest Model: The analysis is inconclusive.")

            # Delete the temporary files after analysis
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)


            fig_waveform, ax_waveform = plt.subplots(figsize=(5, 1))
            ax_waveform.plot(y)
            ax_waveform.axis('off')
            ax_waveform.set_title("Waveform", color='white')
            fig_waveform.patch.set_alpha(0.0)  # Set the figure background to transparent
            ax_waveform.patch.set_alpha(0.0)   # Set the axes background to transparent
            st.pyplot(fig_waveform)
