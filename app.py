import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import plotly.graph_objects as go
import tempfile
import os
import gdown
from audio_recorder_streamlit import audio_recorder

@st.cache_resource
def download_model_if_needed():
    """Download model from Google Drive if not present"""
    model_path = 'models/saved_models/best_model_multilingual.pth'
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.info("🔄 Downloading model for the first time (this takes ~30 seconds)...")
        
        # Create directory
        os.makedirs('models/saved_models', exist_ok=True)
        
        # Your Google Drive file ID
        file_id = "1eolxoEXnUDVc336RLnxotCuwy5maJYqj"  # ← REPLACE THIS!
        
        # Download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            # Download
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.info("Please check the file ID and ensure the file is publicly accessible.")
            st.stop()
    
    return model_path

st.set_page_config(
    page_title="Voice Emotion Recognition",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model architecture (keep same as before)
class CNNLSTM(nn.Module):
    def __init__(self, input_dim=39, num_classes=8, hidden_dim=256, num_layers=2, dropout=0.3):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(256, hidden_dim, num_layers, batch_first=True,
                           bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_dim * 2, 128)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, lengths=None):
        x = self.dropout1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout2(self.relu2(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_combined = torch.cat([h_n[-2, :, :], h_n[-1, :, :]], dim=1)
        x = self.dropout3(self.relu3(self.fc1(h_combined)))
        x = self.fc2(x)
        return x

EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

EMOTION_COLORS = {
    'Neutral': '#95a5a6', 'Calm': '#3498db', 'Happy': '#f1c40f', 'Sad': '#34495e',
    'Angry': '#e74c3c', 'Fearful': '#9b59b6', 'Disgust': '#16a085', 'Surprised': '#e67e22'
}

@st.cache_resource
def load_model():
    model = CNNLSTM()
    model.load_state_dict(torch.load('models/saved_models/best_model_multilingual.pth', map_location='cpu'))
    model.eval()
    return model

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    return features, y, sr

def predict_emotion(model, features):
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    return EMOTIONS[predicted_idx], confidence, probabilities[0].numpy()

def plot_probabilities(probs):
    colors = [EMOTION_COLORS[emotion] for emotion in EMOTIONS]
    fig = go.Figure(data=[
        go.Bar(x=EMOTIONS, y=probs * 100, marker_color=colors,
               text=[f'{p*100:.1f}%' for p in probs], textposition='auto')
    ])
    fig.update_layout(
        title="Emotion Probability Distribution",
        xaxis_title="Emotion", yaxis_title="Probability (%)",
        yaxis_range=[0, 100], height=400, showlegend=False
    )
    return fig

def plot_waveform(audio, sr):
    time = np.arange(0, len(audio)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio, mode='lines',
                            line=dict(color='#3498db', width=1), name='Waveform'))
    fig.update_layout(
        title="Audio Waveform",
        xaxis_title="Time (seconds)", yaxis_title="Amplitude", height=300
    )
    return fig

def process_audio(audio_path, model):
    features, audio, sr = extract_features(audio_path)
    emotion, confidence, probs = predict_emotion(model, features)
    
    st.markdown("---")
    st.header("🎯 Analysis Results")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(label="Detected Emotion", value=emotion,
                 delta=f"{confidence*100:.1f}% confident")
        color = EMOTION_COLORS[emotion]
        st.markdown(
            f'<div style="background-color: {color}; padding: 20px; '
            f'border-radius: 10px; text-align: center; color: white; '
            f'font-size: 24px; font-weight: bold;">{emotion}</div>',
            unsafe_allow_html=True
        )
    
    with col2:
        fig_probs = plot_probabilities(probs)
        st.plotly_chart(fig_probs, use_container_width=True)
    
    st.subheader("📊 Audio Waveform")
    fig_wave = plot_waveform(audio, sr)
    st.plotly_chart(fig_wave, use_container_width=True)
    
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", f"{len(audio)/sr:.2f}s")
    col2.metric("Sample Rate", f"{sr} Hz")
    col3.metric("Samples", f"{len(audio):,}")

def main():
    st.title("🎤 Voice Emotion Recognition System")
    st.markdown("### AI-Powered Emotion Detection from Voice")
    
    # Add navigation hint
    st.info("👈 **Use the sidebar** to navigate between pages:\n- 🏠 Home (Record/Upload)\n- 📁 Batch Processing\n- 📊 Analytics Dashboard")
    
    st.markdown("---")
    
    with st.sidebar:
        st.header("About")
        st.info(
            "CNN-LSTM deep learning model trained on RAVDESS dataset "
            "with **88.19% accuracy**."
        )
        
        st.header("Supported Emotions")
        for emotion in EMOTIONS:
            st.markdown(f"• {emotion}")
        
        st.header("Quick Guide")
        st.markdown(
            "**🏠 Home:** Single file analysis\n\n"
            "**📁 Batch:** Multiple files\n\n"
            "**📊 Analytics:** View statistics"
        )
    
    with st.spinner("Loading model..."):
        model = load_model()
    
    tab1, tab2 = st.tabs(["🎙️ Record Audio", "📁 Upload File"])
    
    with tab1:
        st.subheader("Record Your Voice")
        st.markdown("Click the button below and speak to detect emotion")
        
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e74c3c",
            neutral_color="#3498db",
            icon_name="microphone",
            icon_size="3x",
        )
        
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            with open("temp_recording.wav", "wb") as f:
                f.write(audio_bytes)
            
            with st.spinner("Analyzing..."):
                try:
                    process_audio("temp_recording.wav", model)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists("temp_recording.wav"):
                        os.unlink("temp_recording.wav")
    
    with tab2:
        st.subheader("Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'ogg', 'flac']
        )
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            st.audio(uploaded_file, format='audio/wav')
            
            with st.spinner("Analyzing..."):
                try:
                    process_audio(tmp_path, model)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    os.unlink(tmp_path)
        else:
            st.info("👆 Upload an audio file to get started")

if __name__ == "__main__":
    main()
