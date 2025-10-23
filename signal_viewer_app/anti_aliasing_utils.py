"""
Anti-Aliasing Utility for Audio Enhancement
Based on the provided inference.py
"""

import torch
import torch.nn as nn
import numpy as np
import io
import wave


# ============================================================================
# MODEL ARCHITECTURE (Must match training)
# ============================================================================
class CB(nn.Module):
    def __init__(s, i, o, k=3, st=1, p=1):
        super().__init__()
        s.c = nn.Conv1d(i, o, k, st, p, bias=False)
        s.b = nn.BatchNorm1d(o)
        s.a = nn.LeakyReLU(0.2, inplace=True)

    def forward(s, x):
        return s.a(s.b(s.c(x)))


class RB(nn.Module):
    def __init__(s, c, d=1):
        super().__init__()
        s.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False)
        s.b1 = nn.BatchNorm1d(c)
        s.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d, bias=False)
        s.b2 = nn.BatchNorm1d(c)
        s.a = nn.LeakyReLU(0.2, inplace=True)

    def forward(s, x):
        r = x
        o = s.a(s.b1(s.c1(x)))
        o = s.b2(s.c2(o))
        return s.a(o + r)


class M(nn.Module):
    def __init__(s, h=256, nr=8, nl=3, lh=384):
        super().__init__()
        s.enc = nn.ModuleList([
            CB(1, 32, 7, 1, 3),
            CB(32, 64, 5, 2, 2),
            CB(64, 128, 5, 2, 2),
            CB(128, h, 3, 2, 1)
        ])
        s.res = nn.ModuleList([RB(h, 2 ** i) for i in range(nr)])
        s.lstm = nn.LSTM(h, lh, nl, batch_first=True, bidirectional=True,
                         dropout=0.1 if nl > 1 else 0)
        s.proj = nn.Conv1d(lh * 2, h, 1)
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
            nn.Tanh()
        ])

    def forward(s, x):
        for l in s.enc:
            x = l(x)
        for b in s.res:
            x = b(x)
        bs, ch, ts = x.shape
        x = x.permute(0, 2, 1)
        x, _ = s.lstm(x)
        x = x.permute(0, 2, 1)
        x = s.proj(x)
        for l in s.dec:
            x = l(x)
        return x


class AudioAntiAliaser:
    """Audio Enhancement using Anti-Aliasing Neural Network"""

    def __init__(self, model_path, device='auto', sample_rate=16000,
                 hidden_size=160, num_residual_blocks=5,
                 num_lstm_layers=2, lstm_hidden_size=224):
        """
        Initialize the audio anti-aliaser

        Args:
            model_path: Path to the trained .pth model file
            device: 'cuda', 'cpu', or 'auto' (auto-detect)
            sample_rate: Target sample rate
            hidden_size: Model hidden size (must match training)
            num_residual_blocks: Number of residual blocks
            num_lstm_layers: Number of LSTM layers
            lstm_hidden_size: LSTM hidden size
        """
        # Device setup
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f'[ANTI-ALIAS] Using device: {self.device}')

        self.sample_rate = sample_rate

        # Load model
        print(f'[ANTI-ALIAS] Loading model from: {model_path}')
        self.model = M(hidden_size, num_residual_blocks,
                       num_lstm_layers, lstm_hidden_size).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'epoch' in checkpoint:
                print(f"[ANTI-ALIAS] Model from epoch {checkpoint['epoch']}")
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()
        print('[ANTI-ALIAS] ✅ Model loaded successfully!')

    def process_chunk(self, chunk):
        """Process a single audio chunk"""
        with torch.no_grad():
            # Add batch dimension if needed
            if chunk.dim() == 2:
                chunk = chunk.unsqueeze(0)

            chunk = chunk.to(self.device)

            # Normalize
            max_val = torch.max(torch.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val

            # Process
            enhanced = self.model(chunk)

            # Denormalize
            if max_val > 0:
                enhanced = enhanced * max_val

            return enhanced

    def enhance_audio_array(self, audio_array, sample_rate, chunk_size=40000, overlap=4000):
        """
        Enhance audio from numpy array

        Args:
            audio_array: Numpy array of audio samples
            sample_rate: Sample rate of the audio
            chunk_size: Size of chunks to process
            overlap: Overlap between chunks for smooth transitions

        Returns:
            Enhanced audio as numpy array
        """
        # Convert to torch tensor
        waveform = torch.from_numpy(audio_array).float()

        # Ensure mono and correct shape
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        original_length = waveform.shape[1]

        # Process in chunks with overlap
        hop_size = chunk_size - overlap
        enhanced_chunks = []

        for start in range(0, original_length, hop_size):
            end = min(start + chunk_size, original_length)
            chunk = waveform[:, start:end]

            # Pad if necessary
            if chunk.shape[1] < chunk_size:
                padding = chunk_size - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, padding))

            # Process
            enhanced_chunk = self.process_chunk(chunk)

            # Remove padding
            if end - start < chunk_size:
                enhanced_chunk = enhanced_chunk[:, :, :end - start]

            enhanced_chunks.append(enhanced_chunk.cpu())

        # Combine chunks with crossfade
        if len(enhanced_chunks) == 1:
            enhanced = enhanced_chunks[0].squeeze(0)
        else:
            enhanced = self._combine_chunks(enhanced_chunks, overlap, original_length)

        # Convert back to numpy
        enhanced_np = enhanced.squeeze().numpy()

        # Normalize to prevent clipping
        max_val = np.max(np.abs(enhanced_np))
        if max_val > 0:
            enhanced_np = enhanced_np / max_val * 0.95

        return enhanced_np

    def _combine_chunks(self, chunks, overlap, target_length):
        """Combine overlapping chunks with crossfade"""
        # Initialize output
        output = torch.zeros(1, target_length)
        hop_size = chunks[0].shape[2] - overlap

        for i, chunk in enumerate(chunks):
            start = i * hop_size
            chunk = chunk.squeeze(0)
            chunk_len = chunk.shape[1]
            end = min(start + chunk_len, target_length)
            actual_len = end - start

            if i == 0:
                # First chunk - no fade in
                output[:, start:end] = chunk[:, :actual_len]
            elif i == len(chunks) - 1:
                # Last chunk - fade in only
                fade_in = torch.linspace(0, 1, overlap).unsqueeze(0)
                output[:, start:start + overlap] = (
                        output[:, start:start + overlap] * (1 - fade_in) +
                        chunk[:, :overlap] * fade_in
                )
                output[:, start + overlap:end] = chunk[:, overlap:actual_len]
            else:
                # Middle chunks - crossfade
                fade_in = torch.linspace(0, 1, overlap).unsqueeze(0)
                output[:, start:start + overlap] = (
                        output[:, start:start + overlap] * (1 - fade_in) +
                        chunk[:, :overlap] * fade_in
                )
                output[:, start + overlap:end] = chunk[:, overlap:actual_len]

        return output


def audio_array_to_wav_bytes(audio_data, sample_rate):
    """Convert audio array to WAV file bytes"""
    # Ensure audio is in correct format
    if isinstance(audio_data, list):
        audio_data = np.array(audio_data)

    # Convert to 16-bit PCM
    audio_int16 = (audio_data * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    wav_buffer.seek(0)
    return wav_buffer.getvalue()