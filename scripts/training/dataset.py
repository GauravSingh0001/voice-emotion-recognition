import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class EmotionDataset(Dataset):
    def __init__(self, csv_file, features_dir='data/features/RAVDESS/'):
        """
        Args:
            csv_file: Path to CSV with columns [filename, filepath, emotion]
            features_dir: Directory containing .npy feature files
        """
        self.data = pd.read_csv(csv_file)
        self.features_dir = features_dir
        
        # Emotion to label mapping
        self.emotion_to_label = {
            'neutral': 0,
            'calm': 1,
            'happy': 2,
            'sad': 3,
            'angry': 4,
            'fearful': 5,
            'disgust': 6,
            'surprised': 7
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get filename and emotion
        filename = self.data.iloc[idx]['filename']
        emotion = self.data.iloc[idx]['emotion']
        
        # Load features from .npy file
        feature_path = os.path.join(self.features_dir, filename.replace('.wav', '.npy'))
        features = np.load(feature_path)
        
        # Convert to tensor and transpose to (features, time)
        features = torch.FloatTensor(features).transpose(0, 1)
        
        # Convert emotion to label
        label = self.emotion_to_label[emotion]
        
        return features, label

def pad_collate(batch):
    """
    Custom collate function to handle variable-length sequences
    Pads all sequences to the length of the longest sequence in batch
    """
    # Sort batch by sequence length (descending)
    batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
    
    # Get features and labels
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Get max length
    max_len = features[0].shape[1]
    
    # Pad sequences
    padded_features = []
    lengths = []
    
    for feat in features:
        length = feat.shape[1]
        lengths.append(length)
        
        # Pad if needed
        if length < max_len:
            padding = torch.zeros(feat.shape[0], max_len - length)
            feat = torch.cat([feat, padding], dim=1)
        
        padded_features.append(feat)
    
    # Stack into batch
    features = torch.stack(padded_features)
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    
    return features, labels, lengths
