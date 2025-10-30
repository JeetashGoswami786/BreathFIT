import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import io
import pandas as pd


# --- 1. CONFIGURATION (Updated paths) ---
FIXED_SR = 4000
FIXED_LEN_SAMPLES = 40000
# --- NEW FILE NAMES ---
MODEL_PATH = "breathfit_model_3class.keras"
ENCODER_PATH = "label_encoder_3class.joblib"

# --- 2. LOAD SAVED MODEL AND ENCODER ---
@st.cache_resource
def load_model_and_encoder():
    """Loads the saved Keras model and label encoder."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, le
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, le = load_model_and_encoder()

# --- 3. PREPROCESSING FUNCTION (Same as before) ---
def preprocess_audio(uploaded_file):
    """
    Loads an audio file, pads/truncates it, and converts to a Mel spectrogram.
    """
    try:
        y, sr = librosa.load(uploaded_file, sr=FIXED_SR)
        
        if len(y) > FIXED_LEN_SAMPLES:
            y = y[:FIXED_LEN_SAMPLES]
        elif len(y) < FIXED_LEN_SAMPLES:
            y = np.pad(y, (0, FIXED_LEN_SAMPLES - len(y)), 'constant')
            
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        
        log_spectrogram = log_spectrogram.reshape(1, log_spectrogram.shape[0], log_spectrogram.shape[1], 1)
        
        return log_spectrogram, y # Return both spec and raw wave for plotting
    
    except Exception as e:
        st.error(f"Error processing audio file: {e}")
        return None, None

# --- 4. STREAMLIT APP LAYOUT ---
st.set_page_config(page_title="BreathFIT Analysis Dashboard", layout="wide")
st.title("🫁 BreathFIT: AI Respiratory Screening")
st.write("Upload a `.wav` file of a breath sound to get a 3-class analysis.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Audio")
    uploaded_file = st.file_uploader("Choose a .wav file...", type=["wav", "m4a"])

    if uploaded_file is not None and model is not None:
        st.audio(uploaded_file)
        
        if st.button("Analyze Audio"):
            with st.spinner('Analyzing...'):
                processed_spec, waveform = preprocess_audio(uploaded_file)
                
                if processed_spec is not None:
                    # --- 5. MAKE PREDICTION (NEW 3-CLASS LOGIC) ---
                    # model.predict() now returns an array of 3 probabilities
                    # e.g., [[0.1, 0.05, 0.85]]
                    prediction_probs = model.predict(processed_spec)[0]
                    
                    # Find the index of the highest probability
                    prediction_index = np.argmax(prediction_probs)
                    
                    # Get the class name (e.g., 'wheeze')
                    prediction_class = le.classes_[prediction_index]
                    
                    # Get the probability (the "Risk Score")
                    risk_score = prediction_probs[prediction_index]
                    
                    # --- 6. DISPLAY RESULTS (NEW 3-CLASS LOGIC) ---
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

                        # Display all probabilities (this is your risk/severity breakdown)
                        st.subheader("Probability Breakdown")
                        prob_df = pd.DataFrame(prediction_probs, index=le.classes_, columns=["Probability"])
                        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(prob_df, use_container_width=True)
                        
                        # Plot the waveform and spectrogram
                        st.subheader("Visual Analysis")
                        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                        librosa.display.waveshow(waveform, sr=FIXED_SR, ax=ax[0])
                        ax[0].set_title("Audio Waveform")
                        spec_to_plot = processed_spec.reshape(128, -1) # remove batch/channel dims
                        librosa.display.specshow(spec_to_plot, sr=FIXED_SR, x_axis='time', y_axis='mel', ax=ax[1])
                        ax[1].set_title("Mel Spectrogram")
                        plt.tight_layout()
                        st.pyplot(fig)
