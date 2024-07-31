from io import BytesIO
import streamlit as st
import numpy as np
import os
import joblib
import librosa
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# Loading model
models_path = r"models/"
xgboost_model = joblib.load(os.path.join(models_path, 'XG Boost.pkl'))
gradient_boosting_model = joblib.load(os.path.join(models_path, 'Gradient_Boosting.pkl'))
random_forest_model = joblib.load(os.path.join(models_path, 'Random_Forest.pkl'))



# Function to extract features from the audio file
def extract_features(file_path,  target_sr=16000):
    y, sr = librosa.load(file_path, sr=None)
    
    # Resample to the target sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

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
    #selected_features = features[highly_correlated_indices]

    return features, y, sr

# Function to extract audio from video
def extract_audio_from_video(video_file):
    video = VideoFileClip(video_file)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    return audio_path

# function to predict with a single model
def predict_with_model(audio_file, model):
    features, _, _ = extract_features(audio_file)
    features = features.reshape(1, -1)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        prediction = np.argmax(probabilities)
        confidence = np.max(probabilities)
    else:
        prediction = model.predict(features)[0]
        confidence = None  # If no probabilities, confidence remains undefined
    return prediction, confidence

# Function to predict audio using ensemble of models
def predict_audio(audio_file, models):
    xgboost_model, gradient_boosting_model, random_forest_model = models

    # Get predictions and confidence scores from each model
    xgboost_prediction, xgboost_confidence = predict_with_model(audio_file, xgboost_model)
    gradient_boosting_prediction, gradient_boosting_confidence = predict_with_model(audio_file, gradient_boosting_model)
    random_forest_prediction, random_forest_confidence = predict_with_model(audio_file, random_forest_model)

    # Perform ensemble decision
    predictions = [xgboost_prediction, gradient_boosting_prediction, random_forest_prediction]
    confidences = [xgboost_confidence, gradient_boosting_confidence, random_forest_confidence]

    # Calculate the majority vote
    unique_predictions, counts = np.unique(predictions, return_counts=True)
    majority_prediction = unique_predictions[np.argmax(counts)]
    highest_confidence_model = np.argmax(confidences)

    return majority_prediction, confidences[highest_confidence_model]

# UI
st.set_page_config(layout="wide", page_title="Deepfake Audio Detection", page_icon="")
hide_st_style= """
<style>
#MainMenu {visibility:hidden;}
footer    {visibility:hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar content
audio_source = st.sidebar.selectbox("Select audio source:", ("Audio File", "Video File"))
audio_file = None
temp_files = []  # List to keep track of temporary files

if audio_source == "Audio File":
    audio_file = st.sidebar.file_uploader("Upload your audio file", type=None)
    if audio_file is not None:
        # Workaround to keep file accessible
        audio_bytes = audio_file.read()
        audio_path = "uploaded_audio.mp3"  # Temporary file path
        with open(audio_path, 'wb') as f:
            f.write(audio_bytes)
        st.sidebar.audio(BytesIO(audio_bytes), format="audio/mp3")
        temp_files.append(audio_path)

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
            majority_prediction, highest_confidence = predict_audio(audio_path, [xgboost_model, gradient_boosting_model, random_forest_model])

            # Display the ensemble decision result
            if majority_prediction == 1:
                st.error(f"Your audio is Fake with Confidence of {highest_confidence*100:.2f}%")
            elif majority_prediction == 0:
                st.success(f"Your audio is Real with Confidence of {highest_confidence*100:.2f}%")
            else:
                st.warning("The analysis is inconclusive.")

            # Plotting the waveform
            _, y, sr = extract_features(audio_path)
            fig_waveform, ax_waveform = plt.subplots(figsize=(5, 1))
            ax_waveform.plot(y)
            ax_waveform.axis('off')
            ax_waveform.set_title("Waveform", color='white')
            fig_waveform.patch.set_alpha(0.0)  # Set the figure background to transparent
            ax_waveform.patch.set_alpha(0.0)   # Set the axes background to transparent
            st.pyplot(fig_waveform)

            # Delete the temporary files after analysis
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
