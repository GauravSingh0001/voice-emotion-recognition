import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim=39, num_classes=8, hidden_dim=256, num_layers=2, dropout=0.3):
        """
        CNN-LSTM model for emotion recognition
        
        Args:
            input_dim: Number of MFCC features (default: 39)
            num_classes: Number of emotion classes (default: 8)
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(CNNLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            256, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 128)  # *2 for bidirectional
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x, lengths=None):
        """
        Forward pass
        
        Args:
            x: Input features (batch_size, input_dim, seq_len)
            lengths: Actual lengths of sequences (for packing)
        """
        # CNN layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # Transpose for LSTM (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM layers
        if lengths is not None:
            # Pack padded sequences
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=True
            )
        
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        # Get last layer's forward and backward hidden states
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        
        # Fully connected layers
        x = self.fc1(h_combined)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        
        return x
