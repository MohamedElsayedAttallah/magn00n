"""
Speed Prediction Utilities for Django Integration
Uses CNN14 model for vehicle speed estimation from audio
Matches the exact inference logic from standalone script
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration - CNN14 Model Parameters (MUST MATCH TRAINING)
SR = 32000  # Sample rate (Hz)
DURATION = 10  # Target duration (seconds)
N_MELS = 64  # Number of mel bands
N_FFT = 1024  # FFT window size
HOP_LENGTH = 320  # Hop length for STFT
FMIN = 50  # Minimum frequency
FMAX = 14000  # Maximum frequency


# ========================================================================
# CNN14 PANNS MODEL ARCHITECTURE
# ========================================================================

class ConvBlock(nn.Module):
    """Convolutional block with batch normalization"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        return x


class CNN14(nn.Module):
    """CNN14 model for audio classification, modified for speed regression"""
    def __init__(self, num_outputs=1):
        super(CNN14, self).__init__()
        
        self.bn0 = nn.BatchNorm2d(64)
        
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, 527, bias=True)  # Original classification layer
        
        # Regression head for speed prediction
        self.fc_regression = nn.Linear(2048, num_outputs, bias=True)

    def forward(self, x):
        """
        Input: (batch_size, time_steps, mel_bins) - typically (batch, 1001, 64)
        Output: Predicted speed in km/h
        """
        # Transpose to (batch_size, mel_bins, time_steps)
        x = x.transpose(1, 2)
        
        # Expand to 4D: (batch_size, 1, mel_bins, time_steps)
        x = x.unsqueeze(1)
        
        # Apply bn0 to match pre-trained weights
        x = x.transpose(1, 2)  # (batch, mel_bins, 1, time_steps)
        x = self.bn0(x)
        x = x.transpose(1, 2)  # (batch, 1, mel_bins, time_steps)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        
        # Global average pooling
        x = torch.mean(x, dim=3)
        
        # Max and average pooling over time
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        
        # Regression output - predicted speed
        output = self.fc_regression(embedding)
        
        return output


# ========================================================================
# AUDIO PREPROCESSING FOR CNN14 (EXACT MATCH TO INFERENCE SCRIPT)
# ========================================================================

def load_audio_clip(audio_path, sr=SR, duration=DURATION):
    """
    Load and preprocess audio for CNN14 model
    EXACT MATCH to inference script's load_and_preprocess_audio (step 1)
    
    Args:
        audio_path: Path to audio file
        sr: Sample rate (32000 Hz)
        duration: Target duration in seconds (10s)
    
    Returns:
        Preprocessed audio array of shape (sr * duration,)
    """
    print(f"\n[CNN14] Loading audio: {audio_path}")
    
    # Load audio file at target sample rate
    try:
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        print(f"[CNN14]   ✓ Audio loaded: {len(audio)/sr:.2f} seconds")
    except Exception as e:
        raise ValueError(f"Error loading audio file: {e}")
    
    # Pad or truncate to target length (EXACT MATCH)
    target_length = sr * duration
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)), 
                      mode='constant', constant_values=0)
        print(f"[CNN14]   ℹ Audio padded to {duration} seconds")
    else:
        # Truncate
        audio = audio[:target_length]
        print(f"[CNN14]   ℹ Audio truncated to {duration} seconds")
    
    return audio


def compute_features(audio, sr=SR):
    """
    Compute log-mel spectrogram for CNN14 model
    EXACT MATCH to inference script's load_and_preprocess_audio (step 2)
    
    Args:
        audio: Audio array (sr * duration samples)
        sr: Sample rate (32000 Hz)
    
    Returns:
        Log-mel spectrogram tensor of shape (1, time_steps, n_mels)
        Typically (1, 1001, 64) for 10s @ 32kHz
    """
    # Compute mel spectrogram (EXACT PARAMETERS)
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )
    
    # Convert to log scale (EXACT MATCH)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize (EXACT MATCH - standard normalization)
    log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
    
    # Transpose to (time, mel_bins) (EXACT MATCH)
    log_mel_spec = log_mel_spec.T
    
    # Convert to tensor and add batch dimension (EXACT MATCH)
    mel_spec_tensor = torch.FloatTensor(log_mel_spec).unsqueeze(0)
    
    print(f"[CNN14]   ✓ Mel spectrogram shape: {mel_spec_tensor.shape}")
    
    return mel_spec_tensor


def load_model(model_path, device='cpu'):
    """
    Load CNN14 model from checkpoint
    EXACT MATCH to inference script's load_model function
    
    Args:
        model_path: Path to .pth checkpoint file
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        Loaded CNN14 model in eval mode
    """
    print(f"\n[CNN14] Loading model from: {model_path}")
    
    # Create model
    model = CNN14(num_outputs=1)
    
    # Load checkpoint (EXACT MATCH - weights_only=False for numpy objects)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract model state dict (EXACT MATCH)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights
        model.load_state_dict(state_dict)
        print("[CNN14]   ✓ Model weights loaded successfully!")
        
        # Print checkpoint info if available
        if 'epoch' in checkpoint:
            print(f"[CNN14]   ℹ Trained for {checkpoint['epoch'] + 1} epochs")
        if 'val_mae' in checkpoint:
            print(f"[CNN14]   ℹ Validation MAE: {checkpoint['val_mae']:.4f}")
        if 'val_rmse' in checkpoint:
            print(f"[CNN14]   ℹ Validation RMSE: {checkpoint['val_rmse']:.4f}")
        
    except Exception as e:
        raise ValueError(f"Error loading model checkpoint: {e}")
    
    # Set to evaluation mode (EXACT MATCH)
    model.eval()
    model.to(device)
    
    return model


def predict_speed(model, audio_path, device='cpu'):
    """
    Predict vehicle speed from audio file
    EXACT MATCH to inference script's predict_speed function
    
    Args:
        model: Loaded CNN14 model
        audio_path: Path to audio file
        device: Device to run inference on
    
    Returns:
        Predicted speed in km/h
    """
    # Load and preprocess audio (EXACT MATCH)
    mel_spec = load_audio_clip(audio_path)
    mel_spec = compute_features(mel_spec)
    mel_spec = mel_spec.to(device)
    
    # Run inference (EXACT MATCH)
    print("\n[CNN14] Running inference...")
    with torch.no_grad():
        output = model(mel_spec)
        speed = output.item()
    
    return speed