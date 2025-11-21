import torch
import torch.nn as nn
import librosa
import numpy as np

# Model class (same as training)
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

# Emotion labels
EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']

# Load model
print("Loading model...")
model = CNNLSTM()
model.load_state_dict(torch.load('models/saved_models/best_model_multilingual.pth', map_location='cpu'))
model.eval()
print("✓ Model loaded!")

def predict_emotion(audio_path):
    # Extract features
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta, delta2])
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    # Predict
    with torch.no_grad():
        output = model(features_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item()
    
    return EMOTIONS[predicted_idx], confidence, probabilities[0].numpy()

# Test on a sample from your dataset
test_audio = 'data/raw/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav'  # Change this path
emotion, confidence, probs = predict_emotion(test_audio)

print(f"\n{'='*50}")
print(f"Predicted Emotion: {emotion}")
print(f"Confidence: {confidence*100:.2f}%")
print(f"{'='*50}")
print("\nAll probabilities:")
for i, emo in enumerate(EMOTIONS):
    print(f"{emo:12s}: {probs[i]*100:.2f}%")
