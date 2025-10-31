"""
Anti-Aliasing Audio Enhancement Utility
========================================

This module implements a deep learning-based audio enhancement system that reduces
aliasing artifacts from undersampled audio signals. Uses a hybrid CNN-LSTM architecture
to restore high-frequency information and improve audio quality.

Architecture:
    - Encoder: 4-layer CNN with progressive downsampling
    - Residual blocks: Skip connections with dilated convolutions
    - LSTM: Bidirectional temporal modeling
    - Decoder: 4-layer transposed CNN for signal reconstruction

Model trained to remove aliasing artifacts and enhance audio clarity when dealing
with undersampled or degraded audio signals.
"""

import torch
import torch.nn as nn
import numpy as np
import io
import wave


# ============================================================================
# NEURAL NETWORK COMPONENTS
# ============================================================================

class CB(nn.Module):
    """Convolutional Block: Conv1d → BatchNorm → LeakyReLU
    
    A basic building block combining convolution, batch normalization, and
    activation for stable training and better feature learning.
    """
    def __init__(s, i, o, k=3, st=1, p=1):
        """
        Args:
            i (int): Input channels
            o (int): Output channels
            k (int): Kernel size (default: 3)
            st (int): Stride (default: 1)
            p (int): Padding (default: 1)
        """
        super().__init__()
        s.c = nn.Conv1d(i, o, k, st, p, bias=False)
        s.b = nn.BatchNorm1d(o)
        s.a = nn.LeakyReLU(0.2, inplace=True)

    def forward(s, x):
        return s.a(s.b(s.c(x)))


class RB(nn.Module):
    """Residual Block: Dilated Conv → BatchNorm → Dilated Conv + Skip Connection
    
    Residual blocks with dilated convolutions allow the model to capture
    patterns at different temporal scales without reducing sequence length.
    """
    def __init__(s, c, d=1):
        """
        Args:
            c (int): Number of channels
            d (int): Dilation factor (default: 1, increases receptive field)
        """
        super().__init__()
        s.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False)
        s.b1 = nn.BatchNorm1d(c)
        s.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False)
        s.b2 = nn.BatchNorm1d(c)
        s.a = nn.LeakyReLU(0.2, inplace=True)

    def forward(s, x):
        r = x  # Residual connection (skip)
        o = s.a(s.b1(s.c1(x)))
        o = s.b2(s.c2(o))
        return s.a(o + r)  # Add skip connection


class M(nn.Module):
    """Main Anti-Aliasing Enhancement Model (CNN-LSTM Hybrid)
    
    Architecture flow:
        Input → Encoder → Residual Blocks → LSTM → Projection → Decoder → Output
    
    The model learns to remove aliasing artifacts and restore audio quality
    through progressive feature extraction and temporal modeling.
    """
    def __init__(s, h=256, nr=8, nl=3, lh=384):
        """
        Args:
            h (int): Hidden size for encoder/decoder (default: 256)
            nr (int): Number of residual blocks (default: 8)
            nl (int): Number of LSTM layers (default: 3)
            lh (int): LSTM hidden size (default: 384)
        """
        super().__init__()
        
        # ENCODER: Progressive downsampling (4 layers, /8 total)
        # Input shape: [batch, 1, time]
        s.enc = nn.ModuleList([
            CB(1, 32, 7, 1, 3),        # 1 → 32 channels (no downsample)
            CB(32, 64, 5, 2, 2),       # 32 → 64 channels (/2 time)
            CB(64, 128, 5, 2, 2),      # 64 → 128 channels (/2 time)
            CB(128, h, 3, 2, 1)        # 128 → h channels (/2 time)
        ])
        
        # RESIDUAL BLOCKS: Multi-scale temporal receptive fields
        # Each block has increasing dilation (1, 2, 4, 8, 16, 32, 64, 128)
        s.res = nn.ModuleList([RB(h, 2 ** i) for i in range(nr)])
        
        # LSTM: Bidirectional temporal context modeling
        s.lstm = nn.LSTM(h, lh, nl, batch_first=True, bidirectional=True,
                         dropout=0.1 if nl > 1 else 0)
        
        # PROJECTION: Map LSTM output (lh*2 for bidirectional) back to h channels
        s.proj = nn.Conv1d(lh * 2, h, 1)
        
        # DECODER: Progressive upsampling (4 layers, ×8 total)
        # Reconstruct audio with learned features
        s.dec = nn.ModuleList([
            nn.ConvTranspose1d(h, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 1, 7, 1, 3),
            nn.Tanh()  # Output in range [-1, 1]
        ])

    def forward(s, x):
        """Forward pass through the model
        
        Args:
            x: Input tensor [batch, 1, time_steps]
            
        Returns:
            Enhanced audio tensor [batch, 1, time_steps]
        """
        # ENCODING: Extract features with downsampling
        for l in s.enc:
            x = l(x)
        
        # RESIDUAL PROCESSING: Multi-scale feature refinement
        for b in s.res:
            x = b(x)
        
        # LSTM: Temporal context (bidirectional)
        bs, ch, ts = x.shape
        x = x.permute(0, 2, 1)  # [batch, time, channels]
        x, _ = s.lstm(x)        # Process temporal sequence
        x = x.permute(0, 2, 1)  # [batch, channels, time]
        
        # PROJECT: Return to model dimension (h)
        x = s.proj(x)
        
        # DECODING: Reconstruct with upsampling
        for l in s.dec:
            x = l(x)
        
        return x


# ============================================================================
# MAIN AUDIO ANTI-ALIASER CLASS
# ============================================================================

class AudioAntiAliaser:
    """Audio Enhancement Engine for Aliasing Removal
    
    Processes audio through a trained neural network to remove aliasing artifacts,
    enhance clarity, and restore high-frequency information from undersampled signals.
    
    Handles chunk-based processing with overlapping windows for memory efficiency
    and smooth transitions between processed segments.
    """

    def __init__(self, model_path, device='auto', sample_rate=16000,
                 hidden_size=160, num_residual_blocks=5,
                 num_lstm_layers=2, lstm_hidden_size=224):
        """
        Initialize the audio anti-aliaser
        
        Args:
            model_path (str): Path to the trained PyTorch .pth model file
            device (str): 'cuda', 'cpu', or 'auto' for auto-detection (default: 'auto')
            sample_rate (int): Target sample rate for processing (default: 16000 Hz)
            hidden_size (int): Model hidden size - MUST match training config (default: 160)
            num_residual_blocks (int): Number of residual blocks - MUST match training (default: 5)
            num_lstm_layers (int): Number of LSTM layers - MUST match training (default: 2)
            lstm_hidden_size (int): LSTM hidden size - MUST match training (default: 224)
        
        Raises:
            RuntimeError: If model loading fails or file not found
        """
        # === DEVICE SETUP ===
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f'[ANTI-ALIAS] Using device: {self.device}')

        self.sample_rate = sample_rate

        # === MODEL INITIALIZATION ===
        print(f'[ANTI-ALIAS] Loading model from: {model_path}')
        self.model = M(hidden_size, num_residual_blocks,
                       num_lstm_layers, lstm_hidden_size).to(self.device)

        # === CHECKPOINT LOADING ===
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats (with/without epoch metadata)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"[ANTI-ALIAS] Model from epoch {checkpoint['epoch']}")
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)

        self.model.eval()  # Set to evaluation mode (disables dropout, batchnorm)
        print('[ANTI-ALIAS] ✅ Model loaded successfully!')

    def process_chunk(self, chunk):
        """Process a single audio chunk through the model
        
        Args:
            chunk: Tensor of shape [batch, 1, time] or [1, time]
            
        Returns:
            Enhanced tensor in same shape as input
        """
        with torch.no_grad():  # Disable gradient computation (inference only)
            # Add batch dimension if needed
            if chunk.dim() == 2:
                chunk = chunk.unsqueeze(0)

            chunk = chunk.to(self.device)

            # === NORMALIZATION ===
            # Prevent numerical issues by normalizing to [-1, 1]
            max_val = torch.max(torch.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val

            # === INFERENCE ===
            enhanced = self.model(chunk)

            # === DENORMALIZATION ===
            # Restore original amplitude range
            if max_val > 0:
                enhanced = enhanced * max_val

            return enhanced

    def enhance_audio_array(self, audio_array, sample_rate, chunk_size=40000, overlap=4000):
        """Enhance full audio array with chunk-based processing
        
        Processes audio in overlapping chunks to handle long sequences memory-efficiently
        while maintaining smooth transitions between processed segments.
        
        Args:
            audio_array (np.ndarray): Audio samples (mono or multi-channel)
            sample_rate (int): Sample rate of input audio
            chunk_size (int): Size of chunks to process (default: 40000 samples)
            overlap (int): Overlap between chunks in samples (default: 4000 samples)
            
        Returns:
            np.ndarray: Enhanced audio array (mono, same length as input)
        """
        # === SETUP ===
        # Convert to PyTorch tensor
        waveform = torch.from_numpy(audio_array).float()

        # Ensure mono and correct shape [1, time]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            # Multi-channel → mono (average across channels)
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        original_length = waveform.shape[1]

        # === CHUNK PROCESSING ===
        hop_size = chunk_size - overlap  # Distance between chunk starts
        enhanced_chunks = []

        for start in range(0, original_length, hop_size):
            end = min(start + chunk_size, original_length)
            chunk = waveform[:, start:end]

            # Pad chunk to fixed size if necessary
            if chunk.shape[1] < chunk_size:
                padding = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            # Process chunk through model
            enhanced_chunk = self.process_chunk(chunk)

            # Remove padding from last chunk
            if end - start < chunk_size:
                enhanced_chunk = enhanced_chunk[:, :, :end - start]

            enhanced_chunks.append(enhanced_chunk.cpu())

        # === COMBINE CHUNKS ===
        if len(enhanced_chunks) == 1:
            enhanced = enhanced_chunks[0].squeeze(0)
        else:
            # Combine with crossfading for smooth transitions
            enhanced = self._combine_chunks(enhanced_chunks, overlap, original_length)

        # === FINALIZATION ===
        # Convert back to numpy
        enhanced_np = enhanced.squeeze().numpy()

        # Normalize to prevent clipping (95% of max to leave headroom)
        max_val = np.max(np.abs(enhanced_np))
        if max_val > 0:
            enhanced_np = enhanced_np / max_val * 0.95

        return enhanced_np

    def _combine_chunks(self, chunks, overlap, target_length):
        """Combine overlapping chunks with crossfade blending
        
        Uses linear crossfading in overlap regions to create smooth transitions
        between processed chunks, eliminating discontinuities.
        
        Args:
            chunks (list): List of enhanced audio chunks
            overlap (int): Number of overlapping samples
            target_length (int): Final output length
            
        Returns:
            Tensor: Combined audio [1, target_length]
        """
        # Initialize output buffer
        output = torch.zeros(1, target_length)
        hop_size = chunks[0].shape[2] - overlap

        for i, chunk in enumerate(chunks):
            start = i * hop_size
            chunk = chunk.squeeze(0)
            chunk_len = chunk.shape[1]
            end = min(start + chunk_len, target_length)
            actual_len = end - start

            if i == 0:
                # === FIRST CHUNK ===
                # No fade in, just place directly
                output[:, start:end] = chunk[:, :actual_len]
            elif i == len(chunks) - 1:
                # === LAST CHUNK ===
                # Fade in only (previous chunk already in buffer)
                fade_in = torch.linspace(0, 1, overlap).unsqueeze(0)
                output[:, start:start + overlap] = (
                        output[:, start:start + overlap] * (1 - fade_in) +
                        chunk[:, :overlap] * fade_in
                )
                output[:, start + overlap:end] = chunk[:, overlap:actual_len]
            else:
                # === MIDDLE CHUNKS ===
                # Crossfade: blend old chunk out, new chunk in
                fade_in = torch.linspace(0, 1, overlap).unsqueeze(0)
                output[:, start:start + overlap] = (
                        output[:, start:start + overlap] * (1 - fade_in) +
                        chunk[:, :overlap] * fade_in
                )
                output[:, start + overlap:end] = chunk[:, overlap:actual_len]

        return output


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def audio_array_to_wav_bytes(audio_data, sample_rate):
    """Convert audio array to WAV file bytes
    
    Encodes audio samples as 16-bit PCM WAV format for playback or transmission.
    
    Args:
        audio_data (np.ndarray or list): Audio samples in range [-1, 1]
        sample_rate (int): Sample rate in Hz
        
    Returns:
        bytes: WAV file data (can be written to file or played via browser)
    """
    # Ensure audio is numpy array
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)

    # === CONVERT TO 16-BIT PCM ===
    # Range: [-32768, 32767]
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # === CREATE WAV FILE ===
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)        # Mono audio
        wav_file.setsampwidth(2)        # 16-bit (2 bytes per sample)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer.getvalue()
