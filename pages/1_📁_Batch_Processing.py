import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import tempfile
import os
from datetime import datetime
import gdown

st.set_page_config(page_title="Batch Processing", page_icon="📁", layout="wide")

@st.cache_resource
def download_model_if_needed():
    """Download model from Google Drive if not present"""
    model_path = 'models/saved_models/best_model_multilingual.pth'
    
    if not os.path.exists(model_path):
        st.info("🔄 Downloading model for the first time (30 seconds)...")
        os.makedirs('models/saved_models', exist_ok=True)
        
        file_id = "1eolxoEXnUDVc336RLnxotCuwy5maJYqj"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        try:
            gdown.download(url, model_path, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading model: {e}")
            st.info("Please check the file ID and ensure the file is publicly accessible.")
            st.stop()
    
    return model_path

# Same model class
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

@st.cache_resource
def load_model():
    # Download model if needed (THIS IS THE CRITICAL CHANGE!)
    model_path = download_model_if_needed()
    
    # Load model
    model = CNNLSTM()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    return features, len(y)/sr

def predict_emotion(model, features):
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    return EMOTIONS[predicted_idx], confidence

st.title("📁 Batch Emotion Analysis")
st.markdown("### Process multiple audio files at once")
st.markdown("---")

with st.sidebar:
    st.header("Batch Processing")
    st.info("Upload multiple files and analyze them all at once. Export results as CSV.")
    st.markdown(
        "**Benefits:**\n"
        "- Process many files quickly\n"
        "- Export to CSV/Excel\n"
        "- View statistics"
    )

model = load_model()

uploaded_files = st.file_uploader(
    "Upload multiple audio files",
    type=['wav', 'mp3', 'ogg', 'flac'],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"✓ {len(uploaded_files)} files uploaded")
    
    if st.button("🚀 Start Analysis", type="primary"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}... ({i+1}/{len(uploaded_files)})")
            
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(file.read())
                    tmp_path = tmp_file.name
                
                features, duration = extract_features(tmp_path)
                emotion, confidence = predict_emotion(model, features)
                
                results.append({
                    'File Name': file.name,
                    'Emotion': emotion,
                    'Confidence': f"{confidence*100:.2f}%",
                    'Duration': f"{duration:.2f}s",
                    'Status': '✓'
                })
                os.unlink(tmp_path)
            except Exception as e:
                results.append({
                    'File Name': file.name,
                    'Emotion': 'Error',
                    'Confidence': 'N/A',
                    'Duration': 'N/A',
                    'Status': f'✗ {str(e)[:30]}'
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("✓ Complete!")
        
        st.markdown("---")
        st.header("📊 Results")
        
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True)
        
        # Stats
        successful = df[df['Status'] == '✓']
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(df))
        col2.metric("Success", len(successful))
        col3.metric("Failed", len(df) - len(successful))
        
        # Download
        st.markdown("---")
        df['Analysis Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csv = df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="primary"
        )

else:
    st.info("👆 Upload multiple files to start")
