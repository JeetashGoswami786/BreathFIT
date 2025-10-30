import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time # Used for the "fake" 10-second recording

# --- 1. CONFIGURATION ---
FIXED_SR = 4000
FIXED_LEN_SAMPLES = 40000
MODEL_PATH = "breathfit_model_3class.keras"
ENCODER_PATH = "label_encoder_3class.joblib"

# --- NEW: Define the specific demo audio file ---
# This file MUST be in your GitHub repository, in the same folder as dashboard.py
DEMO_AUDIO_FILE = "BP81_N,N,P L U,33,M.wav"

# --- 2. LOAD SAVED MODEL AND ENCODER (Runs only once) ---
@st.cache_resource
def load_model_and_encoder():
    """Loads the saved Keras model and label encoder."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, le
    except FileNotFoundError:
        st.error(f"Fatal Error: Model or Encoder file not found.")
        st.error(f"Make sure '{MODEL_PATH}' and '{ENCODER_PATH}' are in your GitHub repository.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, le = load_model_and_encoder()

# --- 3. PREPROCESSING FUNCTION (Modified to accept a file path) ---
def preprocess_audio(file_path):
    """
    Loads an audio file from a path, pads/truncates, and creates a Mel spectrogram.
    """
    try:
        # Load the audio file from the local path
        y, sr = librosa.load(file_path, sr=FIXED_SR)
        
        # Pad or Truncate
        if len(y) > FIXED_LEN_SAMPLES:
            y = y[:FIXED_LEN_SAMPLES]
        elif len(y) < FIXED_LEN_SAMPLES:
            y = np.pad(y, (0, FIXED_LEN_SAMPLES - len(y)), 'constant')
            
        # Create Mel Spectrogram
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        # Reshape for the CNN
        log_spectrogram = log_spectrogram.reshape(1, log_spectrogram.shape[0], log_spectrogram.shape[1], 1)
        
        return log_spectrogram, y # Return both spec and raw wave for plotting
    
    except FileNotFoundError:
        st.error(f"FATAL ERROR: Demo file '{DEMO_AUDIO_FILE}' not found.")
        st.error("Please upload this file to your GitHub repository.")
        return None, None
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None

# --- 4. STREAMLIT APP LAYOUT (New Demo Flow) ---
st.set_page_config(page_title="BreathFIT Analysis Dashboard", layout="wide")
st.title("🫁 BreathFIT: AI Respiratory Screening")
st.write("Press 'Start Recording' to begin a 10-second simulated screening.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Live Audio Capture")
    
    # 1. The new "Record" button
    if st.button("Start Recording 🔴"):
        
        # Check if models are loaded before proceeding
        if model is None or le is None:
            st.error("Cannot start. Model files are not loaded.")
        else:
            # --- 2. Fake "Recording" Animation ---
            status_text = st.info("Recording in progress... Please breathe normally.")
            progress_bar = st.progress(0.0, text="Recording: 0%")
            
            # Loop for 10 seconds (100 steps * 0.1s sleep = 10s)
            for i in range(100):
                time.sleep(0.1) 
                progress_percent = (i + 1) / 100.0
                progress_bar.progress(progress_percent, text=f"Recording: {i + 1}%")
            
            progress_bar.empty()
            status_text.info("Recording complete. Analyzing audio...")

            # --- 3. "Analyzing" Animation & Real Analysis ---
            with st.spinner('AI is analyzing the sample...'):
                
                # Load and process the *specific* demo file
                processed_spec, waveform = preprocess_audio(DEMO_AUDIO_FILE)
                
                if processed_spec is not None:
                    # --- 4. MAKE PREDICTION ---
                    prediction_probs = model.predict(processed_spec)[0]
                    prediction_index = np.argmax(prediction_probs)
                    prediction_class = le.classes_[prediction_index]
                    risk_score = prediction_probs[prediction_index]
                    
                    # --- 5. DISPLAY RESULTS (in the second column) ---
                    with col2:
                        st.header("Analysis Result")
                        
                        if prediction_class == 'normal':
                            st.success(f"**Diagnosis: NORMAL**")
                            st.metric(label="Confidence", value=f"{risk_score*100:.1f}%")
                        elif prediction_class == 'crackle':
                            st.error(f"**Diagnosis: CRACKLE Detected**")
                            st.metric(label="Severity / Risk Score", value=f"{risk_score*100:.1f}%")
                        elif prediction_class == 'wheeze':
                            st.warning(f"**Diagnosis: WHEEZE Detected**")
                            st.metric(label="Severity / Risk Score", value=f"{risk_score*100:.1f}%")

                        # Display probability breakdown
                        st.subheader("Probability Breakdown")
                        prob_df = pd.DataFrame(prediction_probs, index=le.classes_, columns=["Probability"])
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(prob_df, use_container_width=True)
                        
                        # Plot the waveform and spectrogram
                        st.subheader("Visual Analysis")
                        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                        librosa.display.waveshow(waveform, sr=FIXED_SR, ax=ax[0])
                        ax[0].set_title("Audio Waveform")
                        spec_to_plot = processed_spec.reshape(128, -1)
                        librosa.display.specshow(spec_to_plot, sr=FIXED_SR, x_axis='time', y_axis='mel', ax=ax[1])
                        ax[1].set_title("Mel Spectrogram")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Also show the audio player for the demo file
                        st.subheader("Demo Audio Sample")
                        st.audio(DEMO_AUDIO_FILE)

            status_text.empty() # Clear the "Analyzing..." text
