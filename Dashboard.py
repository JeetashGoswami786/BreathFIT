import streamlit as st
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import time

# --- 1. CONFIGURATION ---
FIXED_SR = 4000
FIXED_LEN_SAMPLES = 40000
MODEL_PATH = "breathfit_model_3class.keras"
ENCODER_PATH = "label_encoder_3class.joblib"
DEMO_AUDIO_FILE = "BP81_N,N,P L U,33,M.wav"

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="BreathFIT AI Dashboard",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 3. CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    /* Background gradient */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #e0f7fa 0%, #f3e5f5 100%);
        color: #222;
    }

    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #283593;
        text-align: center;
        margin-bottom: 0.3em;
    }

    .sub-title {
        font-size: 1.1rem;
        color: #424242;
        text-align: center;
        margin-bottom: 1.5em;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #1E88E5;
        color: white;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        background-color: #1565C0;
        transform: scale(1.05);
    }

    /* Metric boxes */
    div[data-testid="stMetricValue"] {
        color: #0D47A1 !important;
        font-weight: 700;
        font-size: 1.5rem;
    }

    /* Table Styling */
    .stDataFrame {
        border-radius: 12px;
        background-color: #ffffffcc;
    }

    /* Audio Player */
    audio {
        width: 100%;
        margin-top: 10px;
        border-radius: 8px;
    }

    </style>
""", unsafe_allow_html=True)

# --- 4. LOAD MODEL + ENCODER ---
@st.cache_resource
def load_model_and_encoder():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
        return model, le
    except FileNotFoundError:
        st.error(f"Model or Encoder not found. Ensure '{MODEL_PATH}' and '{ENCODER_PATH}' exist.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, le = load_model_and_encoder()

# --- 5. PREPROCESS AUDIO ---
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=FIXED_SR)
        y = np.pad(y, (0, max(0, FIXED_LEN_SAMPLES - len(y))))[:FIXED_LEN_SAMPLES]
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
        log_spectrogram = log_spectrogram.reshape(1, log_spectrogram.shape[0], log_spectrogram.shape[1], 1)
        return log_spectrogram, y
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None, None

# --- 6. MAIN INTERFACE ---
st.markdown('<div class="main-title">🫁 BreathFIT: AI Respiratory Screening</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Real-time AI-based detection of respiratory anomalies (Normal / Crackle / Wheeze)</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.header("🎙️ Live Audio Capture")
    st.write("Press below to simulate a 10-second breathing test sample.")

    if st.button("Start Recording 🔴"):
        if model is None or le is None:
            st.error("Model not loaded. Please check model files.")
        else:
            status_text = st.info("Recording in progress... Please breathe normally.")
            progress_bar = st.progress(0.0, text="Recording: 0%")
            for i in range(100):
                time.sleep(0.1)
                progress_bar.progress((i + 1) / 100.0, text=f"Recording: {i + 1}%")
            progress_bar.empty()
            status_text.info("Recording complete. Analyzing audio...")

            with st.spinner('🤖 AI is analyzing the breathing pattern...'):
                processed_spec, waveform = preprocess_audio(DEMO_AUDIO_FILE)

                if processed_spec is not None:
                    prediction_probs = model.predict(processed_spec)[0]
                    prediction_index = np.argmax(prediction_probs)
                    prediction_class = le.classes_[prediction_index]
                    risk_score = prediction_probs[prediction_index]

                    with col2:
                        st.header("🧠 AI Diagnosis Result")
                        st.markdown("---")

                        color_map = {
                            'normal': ("#43A047", "✅ Normal Breathing Detected"),
                            'crackle': ("#E53935", "🚨 Crackle Sounds Detected"),
                            'wheeze': ("#FB8C00", "⚠️ Wheeze Sounds Detected")
                        }

                        color, message = color_map.get(prediction_class, ("#1976D2", "Unknown Class"))
                        st.markdown(f"<h3 style='color:{color}'>{message}</h3>", unsafe_allow_html=True)
                        st.metric("AI Confidence", f"{risk_score*100:.1f}%")

                        # Probability Table
                        st.markdown("### 📊 Probability Breakdown")
                        prob_df = pd.DataFrame({
                            "Class": le.classes_,
                            "Probability (%)": [f"{p*100:.2f}%" for p in prediction_probs]
                        })
                        st.dataframe(prob_df, use_container_width=True)

                        # Visualizations
                        st.markdown("### 🔍 Audio Visualization")
                        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
                        librosa.display.waveshow(waveform, sr=FIXED_SR, ax=ax[0], color=color)
                        ax[0].set_title("Waveform", fontsize=12)
                        spec_to_plot = processed_spec.reshape(128, -1)
                        librosa.display.specshow(spec_to_plot, sr=FIXED_SR, x_axis='time', y_axis='mel', ax=ax[1], cmap='magma')
                        ax[1].set_title("Mel Spectrogram", fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Audio Player
                        st.markdown("### 🎧 Listen to Sample Audio")
                        st.audio(DEMO_AUDIO_FILE)

            status_text.empty()

# --- 7. SIDEBAR INFO ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2927/2927347.png", width=100)
    st.markdown("## About BreathFIT")
    st.write("""
    **BreathFIT** is an AI-powered tool for detecting early-stage respiratory anomalies such as:
    - Crackle (possible lung infection)
    - Wheeze (asthma indicators)
    - Normal breathing

    Developed using **TensorFlow**, **Librosa**, and **Streamlit**.
    """)
    st.markdown("---")
    st.markdown("📅 *Demo Version 1.0*")
    st.markdown("👨‍💻 Developed by: **Team FERN AI Lab**")

