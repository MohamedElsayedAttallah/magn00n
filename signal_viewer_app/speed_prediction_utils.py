"""
Speed Prediction Utilities for Django Integration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from pathlib import Path

# Configuration
SR = 11025
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 64
CLIP_LENGTH = 10.0


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """Residual block with SE attention"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        out = F.relu(out)
        return out


class TemporalAttention(nn.Module):
    """Multi-head attention for temporal modeling"""

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x.permute(0, 2, 1)


class AdvancedSpeedPredictor(nn.Module):
    """Advanced architecture with ResNet, SE attention, and multi-scale features"""

    def __init__(self, n_mels=64, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.temporal_attention = TemporalAttention(512, num_heads=8)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.scale1 = nn.Conv1d(512, 256, 1)
        self.scale2 = nn.Conv1d(512, 256, 3, padding=1)
        self.scale3 = nn.Conv1d(512, 256, 5, padding=2)

        fusion_dim = 768
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.speed_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.aux_speed_low = nn.Linear(256, 1)
        self.aux_speed_mid = nn.Linear(256, 1)
        self.f0_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_aux=False):
        x = self.stem(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.avgpool(x4)
        x = x.squeeze(2)
        x = self.temporal_attention(x)

        scale1 = F.relu(self.scale1(x))
        scale2 = F.relu(self.scale2(x))
        scale3 = F.relu(self.scale3(x))

        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)
        features = self.temporal_pool(multi_scale).squeeze(-1)
        fused = self.fusion(features)
        speed = self.speed_head(fused)

        if return_aux:
            aux_low = self.aux_speed_low(fused)
            aux_mid = self.aux_speed_mid(fused)
            f0 = self.f0_head(fused)
            return speed, aux_low, aux_mid, f0
        else:
            return speed


def find_best_checkpoint(checkpoint_dir):
    """Find checkpoint with highest variance - with PyTorch 2.9+ compatibility"""
    from pathlib import Path
    import sys
    import warnings

    checkpoint_dir = Path(checkpoint_dir)

    # Print debug info
    print(f"[CHECKPOINT] Searching in: {checkpoint_dir}")
    print(f"[CHECKPOINT] Directory exists: {checkpoint_dir.exists()}")
    print(f"[CHECKPOINT] Python version: {sys.version}")
    print(f"[CHECKPOINT] PyTorch version: {torch.__version__}")
    print(f"[CHECKPOINT] NumPy version: {np.__version__}")

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    # List all files
    all_files = list(checkpoint_dir.iterdir())
    print(f"[CHECKPOINT] All files in directory: {[f.name for f in all_files]}")

    # Find checkpoint files
    checkpoints = list(checkpoint_dir.glob("best_fold_*.pth"))
    print(f"[CHECKPOINT] Found checkpoint files: {[c.name for c in checkpoints]}")

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint files found in {checkpoint_dir}. "
            f"Looking for files matching pattern 'best_fold_*.pth'. "
            f"Files in directory: {[f.name for f in all_files]}"
        )

    best_ckpt = None
    best_variance = -1
    error_details = []

    print(f"[CHECKPOINT] Evaluating {len(checkpoints)} checkpoint(s)...")

    for ckpt_path in checkpoints:
        try:
            print(f"[CHECKPOINT] Loading: {ckpt_path.name}")

            # Check file size
            file_size = ckpt_path.stat().st_size / (1024 * 1024)  # MB
            print(f"[CHECKPOINT] File size: {file_size:.2f} MB")

            if file_size < 0.1:
                error_msg = f"{ckpt_path.name}: File too small ({file_size:.2f} MB), likely corrupted"
                print(f"[CHECKPOINT] ⚠️ {error_msg}")
                error_details.append(error_msg)
                continue

            # FIXED: PyTorch 2.9+ requires explicit weights_only=False
            # Also add safe globals for numpy compatibility
            checkpoint = None

            # Try Method 1: Load with weights_only=False (explicit)
            try:
                print(f"[CHECKPOINT] Attempting load with weights_only=False...")

                # Add numpy safe globals for PyTorch 2.9+
                if hasattr(torch.serialization, 'add_safe_globals'):
                    # Register numpy types as safe
                    import numpy as np
                    try:
                        # Try to register numpy._core (NumPy 2.0+)
                        if hasattr(np, '_core'):
                            torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                    except:
                        pass

                    try:
                        # Also try numpy.core for older NumPy
                        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                    except:
                        pass

                # Load with weights_only=False to bypass restrictions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    checkpoint = torch.load(
                        ckpt_path,
                        map_location='cpu',
                        weights_only=False
                    )

                print(f"[CHECKPOINT] ✅ Loaded successfully with weights_only=False")

            except Exception as e1:
                print(f"[CHECKPOINT] Load attempt 1 failed: {str(e1)[:200]}")

                # Try Method 2: Use pickle directly (for very old checkpoints)
                try:
                    import pickle
                    print(f"[CHECKPOINT] Attempting load with pickle...")

                    with open(ckpt_path, 'rb') as f:
                        checkpoint = pickle.load(f)

                    print(f"[CHECKPOINT] ✅ Loaded successfully with pickle")

                except Exception as e2:
                    print(f"[CHECKPOINT] Load attempt 2 failed: {str(e2)[:200]}")
                    error_msg = f"{ckpt_path.name}: All load methods failed"
                    error_details.append(error_msg)
                    continue

            # Validate checkpoint structure
            if checkpoint is None:
                error_msg = f"{ckpt_path.name}: Failed to load checkpoint"
                print(f"[CHECKPOINT] ⚠️ {error_msg}")
                error_details.append(error_msg)
                continue

            if not isinstance(checkpoint, dict):
                error_msg = f"{ckpt_path.name}: Invalid checkpoint format (not a dict)"
                print(f"[CHECKPOINT] ⚠️ {error_msg}")
                error_details.append(error_msg)
                continue

            # Check for required keys
            required_keys = ['model_state_dict']
            missing_keys = [k for k in required_keys if k not in checkpoint]

            if missing_keys:
                error_msg = f"{ckpt_path.name}: Missing keys: {missing_keys}"
                print(f"[CHECKPOINT] ⚠️ {error_msg}")
                print(f"[CHECKPOINT] Available keys: {list(checkpoint.keys())}")
                error_details.append(error_msg)
                continue

            # Get variance (default to 0 if not present)
            variance = checkpoint.get('val_variance', 0)
            mae = checkpoint.get('val_mae', 0)

            print(f"[CHECKPOINT] {ckpt_path.name} - Variance: {variance:.2f}, MAE: {mae:.2f}")
            print(f"[CHECKPOINT] Checkpoint keys: {list(checkpoint.keys())}")

            if variance > best_variance:
                best_variance = variance
                best_ckpt = ckpt_path
                print(f"[CHECKPOINT] ✅ New best: {ckpt_path.name}")

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"{ckpt_path.name}: {str(e)}"
            print(f"[CHECKPOINT] ❌ Error loading {ckpt_path.name}:")
            print(error_trace)
            error_details.append(error_msg)
            continue

    if best_ckpt is None:
        detailed_errors = "\n".join([f"  - {err}" for err in error_details])
        raise RuntimeError(
            f"No valid checkpoints found. Tried to load {len(checkpoints)} files but all failed.\n"
            f"Files: {[c.name for c in checkpoints]}\n\n"
            f"Detailed errors:\n{detailed_errors}\n\n"
            f"Possible solutions:\n"
            f"1. Update NumPy: pip install numpy --upgrade\n"
            f"2. Re-train model with current PyTorch/NumPy versions\n"
            f"3. Check PyTorch documentation for version compatibility"
        )

    print(f"[CHECKPOINT] ✅ Selected: {best_ckpt.name} (variance: {best_variance:.2f})")
    return best_ckpt, best_variance

def load_audio_clip(audio_path, sr=SR, clip_length=CLIP_LENGTH):
    """Load and preprocess audio"""
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)

    clip_samples = int(clip_length * sr)
    if len(audio) < clip_samples:
        audio = np.pad(audio, (0, clip_samples - len(audio)))
    else:
        start = (len(audio) - clip_samples) // 2
        audio = audio[start:start + clip_samples]

    return audio


def compute_features(audio, sr=SR):
    """Compute mel spectrogram"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=20,
        fmax=sr // 2
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_spec_db = (mel_spec_db + 80) / 80
    mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)

    return mel_spec_tensor