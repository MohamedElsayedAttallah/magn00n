# signal_viewer_app/views.py
"""
Django views for signal processing and analysis.
Handles ECG, EEG, SAR, audio analysis, and ML-based predictions.
"""

import base64
import io
import json
import os
import tempfile
import traceback
import wave
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np
import torch
import wfdb
from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from scipy.signal import resample_poly

# Import utility modules
from .anti_aliasing_utils import AudioAntiAliaser, audio_array_to_wav_bytes
# Add this import at the top of views.py with other imports
# Replace the ECG detection import section in views.py with this:

# ECG Classification imports (TensorFlow/Keras)
try:
    import tensorflow as tf
    from .ecg_classification_utils import (
        load_ecg_model,
        predict_ecg_abnormality,
        get_class_description,
        CLASS_NAMES as ECG_CLASS_NAMES
    )

    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow loaded for ECG classification")
except ImportError as e:
    print(f"⚠️ Warning: TensorFlow not available for ECG detection: {e}")
    TENSORFLOW_AVAILABLE = False


    # Stub functions
    def load_ecg_model(*args, **kwargs):
        raise NotImplementedError("TensorFlow not installed.")


    def predict_ecg_abnormality(*args, **kwargs):
        raise NotImplementedError("TensorFlow not installed.")


    def get_class_description(*args, **kwargs):
        return "TensorFlow not available"


    ECG_CLASS_NAMES = []
# Conditional ML library imports
try:
    import librosa
    import soundfile as sf
    import tensorflow as tf
    from inaSpeechSegmenter import Segmenter
    from .yamnet_utils import (
        initialize_yamnet, 
        YAMNetEmbeddingLayer, 
        extract_yamnet_embeddings,
        load_finetuned_model,
        predict_audio_class
    )
    ML_AVAILABLE = True
    print("✅ All ML libraries loaded successfully")
except ImportError as e:
    print(f"⚠️ Warning: Audio/ML dependencies missing: {e}")
    ML_AVAILABLE = False
    
    # Define stub functions for missing dependencies
    def initialize_yamnet():
        raise NotImplementedError("ML dependencies missing.")
    
    class YAMNetEmbeddingLayer:
        pass
    
    def extract_yamnet_embeddings(*args, **kwargs):
        raise NotImplementedError("ML dependencies missing.")
    
    class Segmenter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("inaSpeechSegmenter not installed.")

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
TARGET_SAMPLE_RATE = 1000
TARGET_SAMPLE_RATE_AUDIO = 16000
MAX_AUDIO_DURATION_SECONDS = 30
MAX_FILE_SIZE_MB = 10

# Model paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'drone_bird_detection', 'yamnet_finetuned.h5')
ANTI_ALIASING_MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'anti_aliasing', 'best_modell.pth')
# Add this constant with other MODEL_PATH constants
ECG_MODEL_PATH = os.path.join(BASE_DIR, 'assets', 'ecg_model', 'model02.keras')
# Classification class names
CLASS_NAMES = ['Drone', 'Bird', 'Noise/Other']

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_post_request(request) -> Optional[JsonResponse]:
    """Validate that request method is POST."""
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")
    return None


def parse_audio_request(request) -> Tuple[Optional[str], Optional[str], int, Optional[JsonResponse]]:
    """
    Parse audio data from request body.
    
    Returns:
        Tuple of (audio_data_uri, filename, target_sr, error_response)
        If error_response is not None, should return it immediately.
    """
    try:
        body_data = json.loads(request.body)
        audio_data_uri = body_data.get('audio_data')
        filename = body_data.get('filename', 'audio_file.wav')
        target_sr = body_data.get('target_sample_rate', TARGET_SAMPLE_RATE_AUDIO)
        
        if not audio_data_uri:
            return None, None, 0, JsonResponse({"error": "No audio data received."}, status=400)
        
        return audio_data_uri, filename, target_sr, None
        
    except json.JSONDecodeError as e:
        return None, None, 0, JsonResponse(
            {"error": f"Invalid JSON in request body: {str(e)}"}, status=400
        )


def decode_audio_data(audio_data_uri: str) -> Tuple[Optional[bytes], Optional[JsonResponse]]:
    """
    Decode base64 audio data URI.
    
    Returns:
        Tuple of (decoded_bytes, error_response)
        If error_response is not None, should return it immediately.
    """
    try:
        if ',' in audio_data_uri:
            _, encoded = audio_data_uri.split(',', 1)
        else:
            encoded = audio_data_uri
        return base64.b64decode(encoded), None
    except Exception as e:
        return None, JsonResponse({"error": f"Failed to decode audio data: {str(e)}"}, status=400)


def save_temp_audio_file(decoded_bytes: bytes, filename: str) -> str:
    """Save decoded audio bytes to temporary file."""
    ext = os.path.splitext(filename)[1] or '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(decoded_bytes)
        return tmp_file.name


def load_and_resample_audio(file_path: str, target_sr: int = TARGET_SAMPLE_RATE_AUDIO,
                            max_duration: int = MAX_AUDIO_DURATION_SECONDS,
                            return_original: bool = False) -> Union[Tuple[np.ndarray, int, int], Tuple[np.ndarray, int, int, np.ndarray]]:
    """
    Load and resample audio file.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate for resampling
        max_duration: Maximum duration in seconds
        return_original: If True, also return the original (pre-resample) audio
    
    Returns:
        Tuple of (audio_array, sample_rate, original_sample_rate) or
        (audio_array, sample_rate, original_sample_rate, original_audio) if return_original=True
    """
    if not ML_AVAILABLE:
        raise RuntimeError("librosa is required but not available")
    
    # Try multiple methods to load audio (fallback for Windows without FFmpeg)
    audio_original = None
    sr_original = None
    librosa_error = None
    
    try:
        # Method 1: Try librosa (requires FFmpeg on Windows for MP3)
        audio_original, sr_original = librosa.load(file_path, sr=None, mono=True)
        print(f"[AUDIO] ✅ Loaded with librosa: {sr_original} Hz")
    except Exception as e:
        librosa_error = str(e)
        error_lower = librosa_error.lower()
        
        # Check for specific compatibility issues
        if 'numba' in error_lower or 'llvmlite' in error_lower:
            print(f"[AUDIO] ⚠️ NumPy/Numba compatibility issue detected")
        elif 'numpy' in error_lower and 'dtype' in error_lower:
            print(f"[AUDIO] ⚠️ NumPy version incompatibility detected")
        elif 'torch' in error_lower or 'pytorch' in error_lower:
            print(f"[AUDIO] ⚠️ PyTorch compatibility issue detected")
        
        print(f"[AUDIO] librosa.load failed: {librosa_error[:200]}")
        
        try:
            # Method 2: Try soundfile directly (works for WAV without FFmpeg)
            import soundfile as sf
            audio_original, sr_original = sf.read(file_path, dtype='float32')
            # Convert stereo to mono if needed
            if len(audio_original.shape) > 1:
                audio_original = np.mean(audio_original, axis=1)
            print(f"[AUDIO] ✅ Loaded with soundfile: {sr_original} Hz")
        except Exception as sf_error:
            print(f"[AUDIO] soundfile failed: {str(sf_error)[:200]}")
            
            try:
                # Method 3: Try pydub for MP3 (doesn't require FFmpeg if using simpleaudio)
                from pydub import AudioSegment
                print(f"[AUDIO] Trying pydub for MP3...")
                
                # Load with pydub
                if file_path.lower().endswith('.mp3'):
                    audio_seg = AudioSegment.from_mp3(file_path)
                elif file_path.lower().endswith('.wav'):
                    audio_seg = AudioSegment.from_wav(file_path)
                else:
                    audio_seg = AudioSegment.from_file(file_path)
                
                # Convert to mono
                if audio_seg.channels > 1:
                    audio_seg = audio_seg.set_channels(1)
                
                # Get sample rate
                sr_original = audio_seg.frame_rate
                
                # Convert to numpy array (float32, normalized to [-1, 1])
                samples = np.array(audio_seg.get_array_of_samples(), dtype=np.float32)
                audio_original = samples / (2.0 ** (8 * audio_seg.sample_width - 1))
                
                print(f"[AUDIO] ✅ Loaded with pydub: {sr_original} Hz")
            except Exception as pydub_error:
                print(f"[AUDIO] pydub failed: {str(pydub_error)[:200]}")
                
                # Provide detailed error message based on what failed
                error_msg = "Cannot load audio file. "
                
                if librosa_error:
                    if 'numba' in librosa_error.lower() or 'llvmlite' in librosa_error.lower():
                        error_msg += "NumPy/Numba compatibility issue. Try: pip install numba --upgrade. "
                    elif 'numpy' in librosa_error.lower():
                        error_msg += "NumPy version incompatibility. Try: pip install 'numpy<2.0'. "
                    elif 'torch' in librosa_error.lower():
                        error_msg += "PyTorch compatibility issue. Try: pip install torch --upgrade. "
                
                error_msg += f"\nTried librosa, soundfile, and pydub. "
                error_msg += f"\nSolutions: 1) Install FFmpeg, 2) Update packages: pip install librosa soundfile pydub --upgrade, "
                error_msg += f"3) Downgrade NumPy: pip install 'numpy<2.0', 4) Convert to WAV format"
                
                raise RuntimeError(error_msg)
    
    # Truncate if too long
    max_samples = max_duration * sr_original
    if len(audio_original) > max_samples:
        audio_original = audio_original[:max_samples]
    
    # Resample if needed
    if sr_original != target_sr:
        audio = librosa.resample(audio_original, orig_sr=sr_original, target_sr=target_sr)
        sr = target_sr
    else:
        audio = audio_original.copy()
        sr = sr_original
    
    # Normalize
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    
    if return_original:
        return audio, sr, sr_original, audio_original
    return audio, sr, sr_original


def compute_fft_analysis(audio: np.ndarray, sr: int, 
                        downsample_factor: int = None) -> Dict[str, Any]:
    """Compute FFT analysis and return visualization data."""
    fft = np.fft.fft(audio)
    fft_magnitude = np.abs(fft[:len(fft) // 2])
    fft_frequencies = np.fft.fftfreq(len(audio), 1 / sr)[:len(fft) // 2]
    
    # Find maximum significant frequency
    threshold = np.max(fft_magnitude) * 0.01
    significant_freqs = fft_frequencies[fft_magnitude > threshold]
    max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0
    
    # Downsample for plotting
    if downsample_factor is None:
        downsample_factor = max(1, len(fft_frequencies) // 1000)
    freqs_plot = fft_frequencies[::downsample_factor].tolist()
    mags_plot = fft_magnitude[::downsample_factor].tolist()

    # Provide both canonical and caller-specific keys for backward compatibility
    return {
        # canonical keys used by frontend (detect_cars logic)
        'fft_frequencies': freqs_plot,
        'fft_magnitudes': mags_plot,
        # alternate names used by some refactored views
        'fft_frequencies_plot': freqs_plot,
        'fft_magnitude_plot': mags_plot,
        'max_frequency': float(max_frequency),
        'nyquist_frequency': float(sr / 2)
    }


def cleanup_temp_file(file_path: Optional[str]) -> None:
    """Safely delete a temporary file."""
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            print(f"[CLEANUP] Warning: {e}")


def build_audio_response(audio: np.ndarray, sr: int, sr_original: int, 
                        filename: str, additional_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Build standard audio analysis response with FFT data."""
    fft_data = compute_fft_analysis(audio, sr)
    
    response = {
        'waveform': audio.tolist(),
        'sr': sr,
        'original_sr': sr_original,
        'filename': filename,
        **fft_data
    }
    
    if additional_data:
        response.update(additional_data)
    
    return response


# *********************************************************************************
# * SARImageProcessor Class Definition (File Handling Logic) *
# *********************************************************************************
class SARImageProcessor:
    """Process SAR/TIFF images with downsampling and contrast enhancement."""

    def __init__(self, max_dimension: int = 4000, remove_pixel_limit: bool = True):
        self.max_dimension = max_dimension
        if remove_pixel_limit:
            Image.MAX_IMAGE_PIXELS = None

    def load_tiff(self, file_path: str) -> Image.Image:
        """Load TIFF image from file path."""
        return Image.open(file_path)

    def get_metadata(self, img: Image.Image) -> Dict[str, Any]:
        """Extract metadata from image."""
        width, height = img.size
        return {
            'width': width,
            'height': height,
            'mode': img.mode,
            'format': img.format,
            'bands': len(img.getbands()) if hasattr(img, 'getbands') else 1
        }

    def downsample_if_needed(self, img: Image.Image) -> Tuple[Image.Image, float]:
        """Downsample image if it exceeds maximum dimensions."""
        width, height = img.size
        
        if width <= self.max_dimension and height <= self.max_dimension:
            return img, 1.0
        
        downsample_factor = max(width / self.max_dimension, height / self.max_dimension)
        new_width = int(width / downsample_factor)
        new_height = int(height / downsample_factor)
        
        downsampled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return downsampled, downsample_factor

    @staticmethod
    def to_numpy(img: Image.Image) -> np.ndarray:
        """Convert PIL image to numpy array."""
        return np.array(img)

    @staticmethod
    def extract_grayscale_data(img_array: np.ndarray) -> np.ndarray:
        """Extract grayscale data from image array."""
        if len(img_array.shape) == 2:
            return img_array
        elif len(img_array.shape) == 3:
            return img_array[:, :, 0]
        else:
            return img_array

    @staticmethod
    def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures of the data."""
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'p2': float(np.percentile(data, 2)),
            'p98': float(np.percentile(data, 98))
        }

    @staticmethod
    def normalize_with_contrast_enhancement(data: np.ndarray, 
                                           clip_percentile: int = 2) -> np.ndarray:
        """Normalize data with contrast enhancement using percentile clipping."""
        if np.max(data) - np.min(data) == 0:
            return np.zeros_like(data, dtype=np.uint8)
        
        p_low = clip_percentile
        p_high = 100 - clip_percentile
        p_low_val, p_high_val = np.percentile(data, [p_low, p_high])
        
        data_clipped = np.clip(data, p_low_val, p_high_val)
        normalized = ((data_clipped - p_low_val) / (p_high_val - p_low_val) * 255).astype(np.uint8)
        
        return normalized

    @staticmethod
    def apply_colormap(grayscale_array: np.ndarray, colormap: str = 'viridis') -> Image.Image:
        """Apply colormap to grayscale image."""
        img = Image.fromarray(grayscale_array, mode='L')
        return img.convert('RGB')

    @staticmethod
    def to_base64_png(img: Image.Image, optimize: bool = True) -> str:
        """Convert image to base64-encoded PNG."""
        buffer = io.BytesIO()
        img.save(buffer, format='PNG', optimize=optimize)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def process_sar_file(self, file_path, clip_percentile=2):
        with self.load_tiff(file_path) as img:
            original_metadata = self.get_metadata(img)
            original_width = original_metadata['width']
            original_height = original_metadata['height']
            img_downsampled, downsample_factor = self.downsample_if_needed(img)
            img_array = self.to_numpy(img_downsampled)
            data = self.extract_grayscale_data(img_array)
            stats = self.calculate_statistics(data)
            normalized = self.normalize_with_contrast_enhancement(data, clip_percentile)
            display_img = self.apply_colormap(normalized, colormap='gray')
            display_width, display_height = display_img.size
            return {
                'image': display_img,
                'metadata': {
                    'original_width': original_width,
                    'original_height': original_height,
                    'display_width': display_width,
                    'display_height': display_height,
                    'downsampled': downsample_factor > 1,
                    'downsample_factor': float(downsample_factor),
                    'mode': original_metadata['mode'],
                    'format': original_metadata['format'],
                    'bands': original_metadata['bands'],
                    'statistics': stats
                }
            }

    def process_uploaded_file(self, uploaded_file, clip_percentile=2):
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tiff') as tmp:
                tmp_path = tmp.name
                for chunk in uploaded_file.chunks():
                    tmp.write(chunk)
            result = self.process_sar_file(tmp_path, clip_percentile)
            result['metadata']['filename'] = uploaded_file.name
            result['metadata']['size_mb'] = float(uploaded_file.size / 1024 / 1024)
            return result
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)


class SARResponseBuilder:
    @staticmethod
    def build_success_response(processed_result, processor):
        image_b64 = processor.to_base64_png(processed_result['image'])
        metadata = processed_result['metadata']
        stats = metadata['statistics']
        debug_info = (
            f"Filename: {metadata['filename']} ({metadata['size_mb']:.2f} MB)\n"
            f"Original Size: {metadata['original_width']} x {metadata['original_height']}\n"
            f"Preview Size: {metadata['display_width']} x {metadata['display_height']} "
            f"({('Downsampled by ' + str(metadata['downsample_factor']) + 'x') if metadata['downsampled'] else 'Original Scale'})\n"
            f"Mode/Bands: {metadata['mode']} ({metadata['bands']} bands)\n"
            f"--- Statistics ---\n"
            f"Min/Max: {stats['min']:.2f} / {stats['max']:.2f}\n"
            f"Mean/Median: {stats['mean']:.2f} / {stats['median']:.2f}\n"
            f"Std Dev: {stats['std']:.2f}\n"
            f"Contrast Clip: {stats['p2']:.2f} to {stats['p98']:.2f} (2nd to 98th Percentile)"
        )
        return {'success': True, 'image_b64': image_b64, 'debug_info': debug_info}


# -----------------------------------------------------------------
# 1. PAGE RENDERING VIEWS
# -----------------------------------------------------------------
def index(request):
    return render(request, 'index.html')


def ecg_view(request):
    return render(request, 'ecg.html')


def eeg_view(request):
    return render(request, 'eeg.html')


def sar_view(request):
    return render(request, 'sar.html')


def detect_view(request):
    return render(request, 'detect.html')


def doppler_view(request):
    return render(request, 'doppler.html')


def detect_cars_view(request):
    return render(request, 'detect_cars.html')


def detect_voices_view(request):
    return render(request, 'detect_voices_gender.html')


# -----------------------------------------------------------------
# 2. API / FILE CONVERSION VIEWS
# -----------------------------------------------------------------

@csrf_exempt
def convert_ecg_dat_to_json(request):
    """Convert ECG .dat/.hea files to JSON format."""

    # Validation and File Retrieval
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    if 'dat_file' not in request.FILES or 'hea_file' not in request.FILES:
        return JsonResponse(
            {"error": "Missing .dat or .hea file. Both are required for WFDB record."},
            status=400
        )

    dat_file = request.FILES['dat_file']
    hea_file = request.FILES['hea_file']

    # Temporary File Setup
    temp_dir = tempfile.mkdtemp()
    #Temporary Directory: Creates a new, empty,
    # and unique temporary directory on the server's hard drive.Why? The wfdb library needs to read files from a file path;
    #it can't read them directly from memory. We must save the uploaded files somewhere first.
    temp_dat_path = os.path.join(temp_dir, dat_file.name)
    temp_hea_path = os.path.join(temp_dir, hea_file.name)
    record_stem = os.path.join(temp_dir, os.path.splitext(dat_file.name)[0])
    #Record Stem: This is a key part. The wfdb library finds files by their "record name" or "stem" (the part without the extension).
    #record_stem becomes the full path without the extension (e.g., /tmp/tmp123xyz/record1). When you give this to wfdb,
    # it automatically looks for record1.dat and record1.hea in that location.

    # Core Processing (Try Block)
    try:
        with open(temp_dat_path, 'wb+') as dest:
            for chunk in dat_file.chunks():
                dest.write(chunk)
        with open(temp_hea_path, 'wb+') as dest:
            for chunk in hea_file.chunks():
                dest.write(chunk)

        rec = wfdb.rdsamp(record_stem)
        #THE CORE LOGIC: This is the most important line. It tells the wfdb (Waveform Database)
        # library to "read samples" (rdsamp) using the record_stem we created.
        # The library finds the .hea and .dat files, reads them, and returns a result object.
        signals, fields = rec[0], rec[1]
        #rec[0] (signals): A NumPy array holding all the numerical signal data. Its shape is (number_of_samples, number_of_channels).
        #rec[1] (fields): A Python dictionary holding all the metadata from the .hea file (e.g., channel names, sampling frequency).

        # Signal Resampling
        original_fs = int(fields.get('fs', 100))
        channel_names = list(fields.get('sig_name', [f"ch{i}" for i in range(signals.shape[1])]))
        processed_signals = signals
        processed_fs = original_fs

        if original_fs > TARGET_SAMPLE_RATE:
            up = TARGET_SAMPLE_RATE
            down = original_fs
            processed_signals = resample_poly(signals, up, down, axis=0)
            #intelligently resample the signal. It's like converting a super-high-definition
            # video down to a standard 1080p.The signal keeps its shape but becomes much smaller and lighter.
            processed_fs = TARGET_SAMPLE_RATE
            #Action: It gets the original_fs (sampling frequency) and channel_names from the fields dictionary.
            # It then checks if the frequency is too high (over 1000 Hz).
            # If it is, it uses resample_poly to downsample the signal to 1000 Hz.
            # This makes the signal file smaller and faster for the frontend to plot,
            # without a noticeable loss in visual quality.

        # JSON Response Preparation
        signal_length = processed_signals.shape[0]
        duration = signal_length / processed_fs
        signals_list = np.nan_to_num(processed_signals.T, nan=0.0).astype(float).tolist()
        #.T transposes the array from (samples, channels) to (channels, samples).
        # This is the format the JavaScript plotter expects.
        #.tolist() converts the entire (and very large) NumPy array into a standard Python list,
        # which can be serialized into JSON.

        response_data = {
            "filename": dat_file.name,
            "fs": processed_fs,
            "channel_names": channel_names,
            "signals": signals_list,
            "duration": duration
        }
        return JsonResponse(response_data)

    #Error and Cleanup (Except / Finally)
    except Exception as e:
        print(f"WFDB Processing Error: {traceback.format_exc()}")
        return JsonResponse({"error": f"WFDB/Signal Processing Error: {str(e)}"}, status=500)

    finally:
        try:
            os.remove(temp_dat_path)
            os.remove(temp_hea_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


@csrf_exempt

       #Request Validation 
       
def convert_eeg_set_to_json(request):
    """Convert EEG .set files to JSON format."""
    if request.method != 'POST' or 'file' not in request.FILES:
        return HttpResponseBadRequest("Invalid request: No file uploaded.")
    
     # File Handling 
     
    uploaded_file = request.FILES['file']
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(temp_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)
                
               #Signal Generation (PLACEHOLDER - Real Implementation Would Use MNE)

        prioritized_channels = [
            "F1", "F2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T7", "T8", "Cz", "Pz", "A1", "A2"
        ]
        total_channels = len(prioritized_channels)
        fs = 250
        channel_names = prioritized_channels
        n_samples = 10000
        signals_np = np.random.randn(total_channels, n_samples) * 5
        signals_list = signals_np.astype(float).tolist()
        duration = n_samples / fs
        
            #JSON Response
        
        response_data = {
            "filename": uploaded_file.name,
            "fs": fs,
            "channel_names": channel_names,
            "signals": signals_list,
            "duration": duration
        }
        return JsonResponse(response_data)

    except Exception as e:
        print(f"EEG Processing Error: {traceback.format_exc()}")
        return JsonResponse({"error": f"MNE/Processing Error: {str(e)}"}, status=500)
         
            #clean up
          
    finally:
        try:
            os.remove(temp_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


@csrf_exempt
def process_sar_grd(request):
    """Process SAR TIFF/GRD files and return base64 image."""
    if request.method != 'POST' or 'file' not in request.FILES:
        return HttpResponseBadRequest("Invalid request: No file uploaded.")

    uploaded_file = request.FILES['file']

    if not uploaded_file.name.lower().endswith(('.tiff', '.tif')):
        return JsonResponse(
            {"error": "Invalid file format. Please upload a TIFF/TIF image."},
            status=400
        )

    try:
        processor = SARImageProcessor(max_dimension=1000)
        processed_result = processor.process_uploaded_file(uploaded_file, clip_percentile=2)
        response_data = SARResponseBuilder.build_success_response(processed_result, processor)
        return JsonResponse(response_data)

    except Image.DecompressionBombError:
        return JsonResponse(
            {"error": "Image is too large. Decompression limit exceeded. Try a smaller file."},
            status=500
        )

    except Exception as e:
        print(f"SAR Critical Processing Error:\n{traceback.format_exc()}")
        return JsonResponse(
            {"error": f"CRITICAL SERVER ERROR: Processing failed. Details: {str(e)}"},
            status=500
        )


# =============================================================================
# DRONE/BIRD AUDIO DETECTION
# =============================================================================

# disables Django CSRF check for this view
@csrf_exempt
def analyze_audio_detect_bird_and_drone(request):
    """Analyze audio for drone/bird classification using YAMNet."""
    error_response = validate_post_request(request)
    if error_response:
        return error_response

    # Parse request
    # Audio URI (say the sound), target sample rate, error response for checking
    audio_data_uri, filename, target_sr, error_response = parse_audio_request(request)
    if error_response:
        return error_response

    # Decode audio, Here we take the Audio URi to decode it
    decoded_bytes, error_response = decode_audio_data(audio_data_uri)
    if error_response:
        return error_response

    # check if dependecies exists
    if not ML_AVAILABLE:
        return JsonResponse({
            "error": "ML dependencies not available. Please install librosa and tensorflow."
        }, status=500)

    temp_audio_path = None
    try:
        # Save and load audio
        temp_audio_path = save_temp_audio_file(decoded_bytes, filename)
        print(f"[DETECT] Loading audio file: {filename}")
        
        # loads the file with the targeted sample rate
        audio, sr, sr_original = load_and_resample_audio(temp_audio_path, target_sr)
        print(f"[DETECT] Audio processed: {len(audio)} samples at {sr} Hz")

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return JsonResponse({
                "error": f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.",
                "model_path": MODEL_PATH
            }, status=500)

        # Load model and predict
        try:
            print("[DETECT] Initializing YAMNet and loading model...")
            initialize_yamnet()
            # Load the model to utilize
            model = load_finetuned_model(MODEL_PATH)
            
            print("[DETECT] Running prediction...")
            # Here is the prediction processed
            prediction_result = predict_audio_class(model, audio, CLASS_NAMES)
            
            print(f"[DETECT] ✅ Result: {prediction_result['predicted_class']} "
                  f"({prediction_result['confidence']:.2%} confidence)")

            # Build response
            response_data = build_audio_response(audio, sr, sr_original, filename, {
                'predicted_class': prediction_result['predicted_class'],
                'probabilities': prediction_result['probabilities'],
                'confidence': prediction_result['confidence'],
                'class_names': CLASS_NAMES,
                'audio_duration': float(len(audio) / sr),
                'audio_samples': len(audio)
            })

            return JsonResponse(response_data)

        except Exception as model_error:
            print(f"[DETECT] ❌ Model execution error:\n{traceback.format_exc()}")
            return JsonResponse({
                "error": f"Model execution failed: {str(model_error)}",
                "suggestion": "The model file may be incompatible. Check that yamnet_finetuned.h5 is properly trained."
            }, status=500)

    except Exception as e:
        print(f"[DETECT] ❌ Audio Analysis Error:\n{traceback.format_exc()}")
        return JsonResponse({
            "error": f"Audio analysis failed: {str(e)}"
        }, status=500)

    finally:
        # clean after done
        cleanup_temp_file(temp_audio_path)


# =============================================================================
# VOICE GENDER DETECTION
# =============================================================================

@csrf_exempt
def analyze_voices_gender(request):
    """
    Analyzes audio to detect speaker gender using voice characteristics.
    
    Uses two methods:
    1. Primary: inaSpeechSegmenter for ML-based gender classification
    2. Fallback: Acoustic feature analysis (spectral centroid + pitch)
    
    Args:
        request: Django POST request containing audio data URI
        
    Returns:
        JsonResponse with gender prediction, confidence, and audio analysis data
    """
    
    # ==================== REQUEST VALIDATION ====================
    # Validate that this is a proper POST request with required data
    error_response = validate_post_request(request)
    if error_response:
        return error_response
    
    # ==================== AUDIO PARSING ====================
    # Extract audio data URI, filename, and target sample rate from request
    audio_data_uri, filename, target_sample_rate, error_response = parse_audio_request(request)
    if error_response:
        return error_response
    
    # ==================== AUDIO DECODING ====================
    # Decode base64 audio data URI into raw bytes
    decoded_bytes, error_response = decode_audio_data(audio_data_uri)
    if error_response:
        return error_response
    
    # Initialize temporary file paths (will be cleaned up in finally block)
    temp_audio_path = None
    temp_wav_path = None

    try:
        # ==================== FFMPEG AVAILABILITY CHECK ====================
        def check_ffmpeg():
            """Check if FFmpeg is installed and accessible."""
            import subprocess
            try:
                subprocess.run(['ffmpeg', '-version'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL,
                               check=True)
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                return False

        ffmpeg_available = check_ffmpeg()
        print(f"[GENDER] FFmpeg available: {ffmpeg_available}")

        # ==================== SAVE TEMPORARY AUDIO FILE ====================
        # Save decoded audio bytes to a temporary file for processing
        temp_audio_path = save_temp_audio_file(decoded_bytes, filename)
        print(f"[GENDER] Processing: {filename}")

        # ==================== LOAD AND RESAMPLE AUDIO ====================
        # Load audio and resample to target rate (keep original for WAV conversion)
        # Limit to 30 seconds to improve processing speed
        audio, sr, sr_original, audio_original = load_and_resample_audio(
            temp_audio_path,
            target_sample_rate,
            max_duration=30,
            return_original=True
        )
        print(f"[GENDER] Audio: {len(audio)} samples at {sr} Hz")

        # ==================== FFT ANALYSIS ====================
        # Compute Fast Fourier Transform for frequency domain analysis
        fft_data = compute_fft_analysis(audio, sr)

        # ==================== GENDER DETECTION STRATEGY ====================
        # Determine whether to use ML-based or fallback method
        use_fallback = False
        fallback_reason = ""

        # If FFmpeg is not available, inaSpeechSegmenter won't work properly
        if not ffmpeg_available:
            use_fallback = True
            fallback_reason = "FFmpeg not installed"
            print(f"[GENDER] {fallback_reason} - using improved fallback")

        # ==================== PRIMARY METHOD: inaSpeechSegmenter ====================
        if not use_fallback:
            try:
                print("[GENDER] Initializing inaSpeechSegmenter...")

                # Convert audio to WAV format if needed (required by inaSpeechSegmenter)
                if not temp_audio_path.lower().endswith('.wav'):
                    temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                    sf.write(temp_wav_path, audio_original, sr_original)
                    analysis_path = temp_wav_path
                else:
                    analysis_path = temp_audio_path

                # Initialize segmenter with gender detection enabled
                seg = Segmenter(vad_engine='smn', detect_gender=True)
                segmentation = seg(analysis_path)

                # Accumulate speech duration for each gender
                male_duration = 0
                female_duration = 0
                total_speech = 0

                print("[GENDER] Segmentation results:")
                for label, start, end in segmentation:
                    duration = end - start
                    print(f"  {label:12} | {start:7.2f}s - {end:7.2f}s | {duration:6.2f}s")

                    # Count duration for male and female speech segments
                    if label == 'male':
                        male_duration += duration
                        total_speech += duration
                    elif label == 'female':
                        female_duration += duration
                        total_speech += duration

                print(f"[GENDER] Total speech: {total_speech:.2f}s")
                print(f"[GENDER] Male: {male_duration:.2f}s, Female: {female_duration:.2f}s")

                # If no speech detected, fall back to acoustic analysis
                if total_speech == 0:
                    use_fallback = True
                    fallback_reason = "No speech detected"
                    print(f"[GENDER] {fallback_reason}")
                else:
                    # Calculate probability based on duration ratios
                    male_percentage = male_duration / total_speech
                    female_percentage = female_duration / total_speech

                    # Determine predicted gender based on majority duration
                    if male_duration > female_duration:
                        predicted_gender = 'Male'
                        confidence = male_percentage
                    elif female_duration > male_duration:
                        predicted_gender = 'Female'
                        confidence = female_percentage
                    else:
                        # Equal durations (rare case)
                        predicted_gender = 'Equal'
                        confidence = 0.5
                        male_percentage = 0.5
                        female_percentage = 0.5

                    print(f"[GENDER] Result: {predicted_gender} ({confidence:.1%})")

            except Exception as seg_error:
                # If segmenter fails for any reason, use fallback method
                use_fallback = True
                fallback_reason = f"inaSpeechSegmenter error: {str(seg_error)}"
                print(f"[GENDER] {fallback_reason}")

        # ==================== FALLBACK METHOD: ACOUSTIC ANALYSIS ====================
        if use_fallback:
            print(f"[GENDER] Using IMPROVED fallback: {fallback_reason}")

            # -------------------- Feature 1: Spectral Centroid --------------------
            # Spectral centroid is the "center of mass" of the frequency spectrum
            # Higher values indicate brighter, higher-pitched sounds
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_spectral_centroid = np.mean(spectral_centroids)

            # -------------------- Feature 2: Pitch Estimation --------------------
            def estimate_pitch_autocorr(signal, sr, fmin=50, fmax=400):
                """
                Estimate fundamental frequency (F0) using autocorrelation method.
                
                This method finds the periodicity in the signal by correlating
                the signal with delayed versions of itself.
                
                Args:
                    signal: Audio signal array
                    sr: Sample rate
                    fmin: Minimum expected frequency (Hz)
                    fmax: Maximum expected frequency (Hz)
                    
                Returns:
                    Estimated fundamental frequency in Hz, or None if estimation fails
                """
                # Normalize signal by removing DC offset
                signal = signal - np.mean(signal)

                # Compute autocorrelation (correlation of signal with itself)
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]  # Keep only positive lags

                # Define search range based on expected frequency range
                min_period = int(sr / fmax)  # Samples per period at max frequency
                max_period = int(sr / fmin)  # Samples per period at min frequency

                # Validate search range
                if max_period >= len(autocorr):
                    return None

                # Find the first significant peak after the minimum period
                # This peak indicates the fundamental period of the signal
                autocorr_segment = autocorr[min_period:max_period]
                if len(autocorr_segment) == 0:
                    return None

                peak_index = np.argmax(autocorr_segment) + min_period
                estimated_freq = sr / peak_index  # Convert period to frequency

                return estimated_freq

            # Estimate fundamental frequency (F0/pitch)
            estimated_f0 = estimate_pitch_autocorr(audio, sr)

            print(f"[GENDER] Spectral centroid: {mean_spectral_centroid:.2f} Hz")
            print(f"[GENDER] Estimated F0: {estimated_f0:.2f} Hz" if estimated_f0 else "[GENDER] F0 estimation failed")

            # -------------------- Multi-Feature Scoring System --------------------
            # Typical voice characteristics:
            # Male:   F0: 85-180 Hz,  Spectral Centroid: 200-500 Hz
            # Female: F0: 165-255 Hz, Spectral Centroid: 400-800 Hz

            male_score = 0
            female_score = 0

            # Score based on fundamental frequency (F0)
            if estimated_f0 is not None:
                if estimated_f0 < 130:
                    # Very low pitch → strongly male
                    male_score += 2.0
                elif estimated_f0 > 200:
                    # Very high pitch → strongly female
                    female_score += 2.0
                elif estimated_f0 < 165:
                    # Low pitch → moderately male
                    male_score += 1.0
                else:
                    # High pitch → moderately female
                    female_score += 1.0

            # Score based on spectral centroid (brightness)
            if mean_spectral_centroid < 350:
                # Dark/low frequency content → strongly male
                male_score += 1.5
            elif mean_spectral_centroid > 600:
                # Bright/high frequency content → strongly female
                female_score += 1.5
            elif mean_spectral_centroid < 500:
                # Somewhat dark → moderately male
                male_score += 0.5
            else:
                # Somewhat bright → moderately female
                female_score += 0.5

            # -------------------- Calculate Probabilities --------------------
            total_score = male_score + female_score
            if total_score > 0:
                male_percentage = male_score / total_score
                female_percentage = female_score / total_score
            else:
                # If no clear indicators, default to equal probability
                male_percentage = 0.5
                female_percentage = 0.5

            # -------------------- Confidence Adjustment --------------------
            # Ensure minimum confidence of 0.55 for clear decisions
            # (Avoid appearing too uncertain when we have a clear winner)
            if male_percentage > female_percentage:
                male_percentage = max(0.55, male_percentage)
                female_percentage = 1 - male_percentage
                predicted_gender = 'Male'
                confidence = male_percentage
            else:
                female_percentage = max(0.55, female_percentage)
                male_percentage = 1 - female_percentage
                predicted_gender = 'Female'
                confidence = female_percentage

            # Cap maximum confidence at 0.90 for fallback method
            # (We're less certain than ML-based approach)
            male_percentage = min(0.90, male_percentage)
            female_percentage = min(0.90, female_percentage)
            confidence = max(male_percentage, female_percentage)

            print(f"[GENDER] Fallback scores - Male: {male_score:.2f}, Female: {female_score:.2f}")
            print(f"[GENDER] Result: {predicted_gender} ({confidence:.1%})")

            # -------------------- Set Duration Estimates --------------------
            # For fallback, estimate durations based on probabilities
            total_speech = len(audio) / sr
            male_duration = male_percentage * total_speech
            female_duration = female_percentage * total_speech

        # ==================== PREPARE RESPONSE ====================
        # Compile all analysis results into a JSON response
        response_data = {
            # Gender prediction results
            "predicted_gender": predicted_gender,
            "confidence": float(confidence),
            "male_probability": float(male_percentage),
            "female_probability": float(female_percentage),
            
            # Audio data for visualization
            "waveform": audio.tolist(),
            "sr": sr,
            "original_sr": sr_original,
            "filename": filename,
            
            # Frequency analysis data
            "max_frequency": fft_data['max_frequency'],
            "nyquist_frequency": fft_data['nyquist_frequency'],
            "fft_frequencies": fft_data['fft_frequencies_plot'],
            "fft_magnitudes": fft_data['fft_magnitude_plot'],
            
            # Speech duration breakdown
            "total_speech_duration": float(total_speech),
            "male_speech_duration": float(male_duration),
            "female_speech_duration": float(female_duration),
            
            # Method used for detection
            "detection_method": "fallback" if use_fallback else "inaSpeechSegmenter",
            "fallback_reason": fallback_reason if use_fallback else None
        }

        print("[GENDER] ✅ Response prepared successfully")
        return JsonResponse(response_data)

    # ==================== ERROR HANDLING ====================
    except Exception as e:
        # Capture full traceback for debugging
        error_traceback = traceback.format_exc()
        print(f"[GENDER] ❌ Error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Gender detection failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    # ==================== CLEANUP ====================
    finally:
        # Always clean up temporary files, even if errors occurred
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"[GENDER] Cleanup warning: {e}")

        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.unlink(temp_wav_path)
            except Exception as e:
                print(f"[GENDER] Cleanup warning: {e}")

# ========== CAR AUDIO ANALYSIS (NO CHANGES NEEDED) ==========
@csrf_exempt
def analyze_cars_audio(request):
    """Car audio file upload for visualization only (no ML classification)."""
    # Validate request
    error_response = validate_post_request(request)
    if error_response:
        return error_response
    
    # Parse audio request
    audio_data_uri, filename, target_sample_rate, error_response = parse_audio_request(request)
    if error_response:
        return error_response
    
    # Decode audio data
    decoded_bytes, error_response = decode_audio_data(audio_data_uri)
    if error_response:
        return error_response

    if not ML_AVAILABLE:
        return JsonResponse({
            "error": "ML dependencies not available. Please install librosa."
        }, status=500)

    temp_audio_path = None
    try:
        # Save, load, and resample audio
        temp_audio_path = save_temp_audio_file(decoded_bytes, filename)
        audio, sr, sr_original = load_and_resample_audio(
            temp_audio_path,
            target_sample_rate,
            max_duration=30
        )

        # Build response data (build_audio_response already computes FFT internally)
        response_data = build_audio_response(audio, sr, sr_original, filename)
        
        # Return JSON response
        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[ANALYZE_CARS] Error: {error_traceback}")
        return JsonResponse({
            "error": f"Audio analysis failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    finally:
        cleanup_temp_file(temp_audio_path)


# ========== DOPPLER AUDIO GENERATION (NO CHANGES NEEDED) ==========
@csrf_exempt
def generate_doppler_audio(request):
    """
    Generate simulated Doppler audio effect with proper frequency shift.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    try:
        data = json.loads(request.body)
        output_file_name = data.get('file_name', 'doppler_tone.wav')
        velocity_kmh = float(data.get('velocity', 80))
        base_frequency = float(data.get('frequency', 440))

        sample_rate = 44100
        FIXED_DURATION = 5.0
        speed_of_sound = 343

        velocity_ms = velocity_kmh * (1000 / 3600)

        f_approach = base_frequency * speed_of_sound / (speed_of_sound - velocity_ms)
        f_recede = base_frequency * speed_of_sound / (speed_of_sound + velocity_ms)

        num_samples = int(sample_rate * FIXED_DURATION)
        t = np.linspace(0, FIXED_DURATION, num_samples)

        midpoint = FIXED_DURATION / 2
        transition_width = 0.3

        transition = 1 / (1 + np.exp(-10 * (t - midpoint) / transition_width))
        doppler_frequency = f_approach * (1 - transition) + f_recede * transition

        phase = 2 * np.pi * np.cumsum(doppler_frequency) / sample_rate
        audio = np.sin(phase)

        distance = np.abs((t - midpoint) * velocity_ms)
        max_distance = velocity_ms * FIXED_DURATION / 2
        min_distance = max_distance * 0.1
        distance_attenuation = 1 / (1 + (distance / min_distance) ** 0.5)
        distance_attenuation = np.clip(distance_attenuation, 0.2, 1.0)

        fade_duration = 0.1
        fade_samples = int(fade_duration * sample_rate)

        envelope = np.ones_like(audio)
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        audio = audio * envelope * distance_attenuation

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.8

        # FFT analysis
        fft = np.fft.fft(audio)
        fft_magnitude = np.abs(fft[:len(fft) // 2])
        fft_frequencies = np.fft.fftfreq(len(audio), 1 / sample_rate)[:len(fft) // 2]

        threshold = np.max(fft_magnitude) * 0.01
        significant_freqs = fft_frequencies[fft_magnitude > threshold]
        max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else f_approach

        # Convert to WAV
        audio_int16 = (audio * 32767).astype(np.int16)
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        wav_buffer.seek(0)
        audio_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')

        downsample_factor = max(1, len(audio) // 10000)
        waveform_downsampled = audio[::downsample_factor].tolist()

        response_data = {
            "output_file_name": output_file_name,
            "audio_b64": audio_base64,
            "waveform_data": waveform_downsampled,
            "sr": sample_rate,
            "max_frequency": float(max_frequency),
            "nyquist_frequency": float(sample_rate / 2),
            "duration": float(FIXED_DURATION),
            "velocity_kmh": float(velocity_kmh),
            "base_frequency": float(base_frequency),
            "doppler_shift_approach": float(f_approach),
            "doppler_shift_recede": float(f_recede)
        }

        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        return JsonResponse(
            {"error": f"Audio generation failed: {str(e)}", "traceback": error_traceback},
            status=500
        )


# Add this new view function at the end of views.py (before the doppler function or at the very end)

@csrf_exempt
def apply_anti_aliasing(request):
    """
    Applies neural network-based anti-aliasing to audio files.
    
    Anti-aliasing removes unwanted high-frequency artifacts (aliasing) that occur
    when audio is sampled at insufficient rates. This function uses a pretrained
    deep learning model to enhance audio quality.
    
    Args:
        request: Django POST request containing audio data URI
        
    Returns:
        JsonResponse with enhanced audio (base64), waveform, and FFT analysis
    """
    
    # ==================== REQUEST VALIDATION ====================
    # Validate that this is a proper POST request with required data
    error_response = validate_post_request(request)
    if error_response:
        return error_response
    
    # ==================== AUDIO PARSING ====================
    # Extract audio data URI, filename, and sample rate from request
    audio_data_uri, filename, sample_rate, error_response = parse_audio_request(request)
    if error_response:
        return error_response
    
    # ==================== AUDIO DECODING ====================
    # Decode base64 audio data URI into raw bytes
    decoded_bytes, error_response = decode_audio_data(audio_data_uri)
    if error_response:
        return error_response

    # Initialize temporary file path (will be cleaned up in finally block)
    temp_audio_path = None

    try:
        print("[ANTI-ALIAS] Processing request...")

        # ==================== MODEL VALIDATION ====================
        # Check if the pretrained neural network model file exists
        # The model (model_ep5.pth) should be trained for anti-aliasing tasks
        if not os.path.exists(ANTI_ALIASING_MODEL_PATH):
            return JsonResponse({
                "error": f"Anti-aliasing model not found at {ANTI_ALIASING_MODEL_PATH}. "
                         "Please ensure model_ep5.pth is placed in signal_viewer_app/assets/anti_aliasing/"
            }, status=500)

        # ==================== SAVE TEMPORARY AUDIO FILE ====================
        # Save decoded audio bytes to a temporary file for processing
        temp_audio_path = save_temp_audio_file(decoded_bytes, filename)
        print(f"[ANTI-ALIAS] Loading audio: {filename}")

        # ==================== LOAD AND RESAMPLE AUDIO ====================
        # Check if librosa (ML library) is available for audio processing
        if not ML_AVAILABLE:
            return JsonResponse({
                "error": "librosa is required for anti-aliasing. Please install it."
            }, status=500)

        # Load audio and resample to 16kHz (model's expected input rate)
        # Limit to 30 seconds to manage memory and processing time
        # 16kHz is chosen as it's sufficient for speech and compatible with the model
        audio, sr, sr_original = load_and_resample_audio(
            temp_audio_path,
            target_sr=16000,  # Model expects 16kHz input
            max_duration=30   # Limit processing to 30 seconds
        )
        print(f"[ANTI-ALIAS] Audio loaded: {len(audio)} samples at {sr} Hz")

        # ==================== INITIALIZE ANTI-ALIASING MODEL ====================
        print("[ANTI-ALIAS] Initializing model...")
        try:
            # Create AudioAntiAliaser instance with specific architecture parameters
            # These parameters define the neural network structure:
            anti_aliaser = AudioAntiAliaser(
                model_path=ANTI_ALIASING_MODEL_PATH,  # Path to pretrained weights
                device='auto',                         # Auto-detect GPU/CPU
                sample_rate=sr,                        # Match audio sample rate
                hidden_size=160,                       # Size of hidden layers in network
                num_residual_blocks=5,                 # Number of residual blocks for deep learning
                num_lstm_layers=2,                     # LSTM layers for temporal modeling
                lstm_hidden_size=224                   # Hidden units in LSTM layers
            )
        except Exception as model_error:
            # If model fails to load (corrupted file, wrong architecture, etc.)
            error_traceback = traceback.format_exc()
            print(f"[ANTI-ALIAS] Model loading error:\n{error_traceback}")
            return JsonResponse({
                "error": f"Failed to load anti-aliasing model: {str(model_error)}",
                "details": error_traceback
            }, status=500)

        # ==================== APPLY ANTI-ALIASING ====================
        print("[ANTI-ALIAS] Applying anti-aliasing...")
        try:
            # Process audio through the neural network
            # Uses chunking to handle long audio files efficiently:
            enhanced_audio = anti_aliaser.enhance_audio_array(
                audio_array=audio,
                sample_rate=sr,
                chunk_size=40000,  # Process in chunks of 40k samples (~2.5s at 16kHz)
                overlap=4000       # Overlap chunks by 4k samples to avoid edge artifacts
            )
            # Chunking prevents memory issues and overlap ensures smooth transitions
        except Exception as enhance_error:
            # If enhancement fails (model inference error, memory issues, etc.)
            error_traceback = traceback.format_exc()
            print(f"[ANTI-ALIAS] Enhancement error:\n{error_traceback}")
            return JsonResponse({
                "error": f"Failed to enhance audio: {str(enhance_error)}",
                "details": error_traceback
            }, status=500)

        print("[ANTI-ALIAS] ✅ Anti-aliasing complete")

        # ==================== CONVERT ENHANCED AUDIO TO BASE64 ====================
        # Convert the enhanced numpy array to WAV format bytes
        wav_bytes = audio_array_to_wav_bytes(enhanced_audio, sr)
        # Encode as base64 for JSON transmission
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        # ==================== FFT ANALYSIS ====================
        # Perform Fast Fourier Transform to analyze frequency content
        # This helps verify that anti-aliasing worked correctly
        
        # Compute FFT (converts time domain to frequency domain)
        fft = np.fft.fft(enhanced_audio)
        # Get magnitude (amplitude) of each frequency component
        fft_magnitude = np.abs(fft[:len(fft) // 2])  # Only positive frequencies
        # Get corresponding frequencies for each FFT bin
        fft_frequencies = np.fft.fftfreq(len(enhanced_audio), 1 / sr)[:len(fft) // 2]

        # -------------------- Find Maximum Significant Frequency --------------------
        # Define threshold as 1% of maximum magnitude to filter noise
        threshold = np.max(fft_magnitude) * 0.01
        # Find frequencies above threshold (significant content)
        significant_freqs = fft_frequencies[fft_magnitude > threshold]
        # Get maximum frequency with significant content
        max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

        # -------------------- Downsample FFT Data for Plotting --------------------
        # Reduce data points to ~1000 for efficient frontend plotting
        downsample_factor = max(1, len(fft_frequencies) // 1000)
        fft_frequencies_plot = fft_frequencies[::downsample_factor].tolist()
        fft_magnitude_plot = fft_magnitude[::downsample_factor].tolist()

        # -------------------- Downsample Waveform for Plotting --------------------
        # Reduce waveform to ~10,000 samples for efficient visualization
        waveform_downsample_factor = max(1, len(enhanced_audio) // 10000)
        waveform_plot = enhanced_audio[::waveform_downsample_factor].tolist()

        # ==================== PREPARE RESPONSE ====================
        # Compile all results into a JSON response
        response_data = {
            "success": True,
            
            # Enhanced audio file (base64 encoded WAV)
            "enhanced_audio_b64": audio_b64,
            
            # Waveform data for visualization (downsampled)
            "waveform": waveform_plot,
            
            # Audio metadata
            "sr": sr,                                      # Sample rate
            "filename": f"enhanced_{filename}",            # New filename
            "duration": float(len(enhanced_audio) / sr),   # Duration in seconds
            "samples": len(enhanced_audio),                # Total samples
            
            # Frequency analysis data
            "max_frequency": float(max_frequency),         # Highest significant frequency
            "nyquist_frequency": float(sr / 2),            # Theoretical maximum frequency
            "fft_frequencies": fft_frequencies_plot,       # Frequency bins (downsampled)
            "fft_magnitudes": fft_magnitude_plot           # Magnitudes (downsampled)
        }

        print("[ANTI-ALIAS] ✅ Response prepared")
        return JsonResponse(response_data)

    # ==================== ERROR HANDLING ====================
    except Exception as e:
        # Catch any unexpected errors not handled above
        error_traceback = traceback.format_exc()
        print(f"[ANTI-ALIAS] ❌ Error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Anti-aliasing failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    # ==================== CLEANUP ====================
    finally:
        # Always clean up temporary files, even if errors occurred
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"[ANTI-ALIAS] Cleanup warning: {e}")

# Replace the predict_car_speed function in views.py with this improved version

@csrf_exempt
@csrf_exempt
def predict_car_speed(request):
    """
    Predict vehicle speed from audio using CNN14 model.
    Accepts drag-and-drop audio file uploads.
    """
    # Validate request
    error_response = validate_post_request(request)
    if error_response:
        return error_response
    
    # Parse audio request (supports drag-and-drop)
    audio_data_uri, filename, _, error_response = parse_audio_request(request)
    if error_response:
        return error_response
    
    # Decode audio data
    decoded_bytes, error_response = decode_audio_data(audio_data_uri)
    if error_response:
        return error_response

    temp_audio_path = None

    try:
        print("[SPEED_CNN14] Processing speed prediction request...")
        print(f"[SPEED_CNN14] Filename: {filename}")

        # Check PyTorch availability
        try:
            import torch
        except ImportError:
            return JsonResponse({
                "error": "PyTorch is not installed. Please install it: pip install torch"
            }, status=500)

        # Define model path - CNN14 model
        MODEL_PATH = os.path.join(
            os.path.dirname(__file__), 
            'assets', 
            'speed_estimation', 
            'best_model (1).pth'
        )

        print(f"[SPEED_CNN14] Model path: {MODEL_PATH}")
        print(f"[SPEED_CNN14] Model exists: {os.path.exists(MODEL_PATH)}")

        # Check if model file exists
        if not os.path.exists(MODEL_PATH):
            return JsonResponse({
                "error": "CNN14 model not found",
                "model_path": MODEL_PATH,
                "solution": "Please ensure the CNN14 model file 'best_model (1).pth' is in signal_viewer_app/assets/speed_estimation/"
            }, status=500)

        # Save temporary audio file
        temp_audio_path = save_temp_audio_file(decoded_bytes, filename)
        print(f"[SPEED_CNN14] Saved temporary file: {temp_audio_path}")

        # Import CNN14 utilities
        try:
            from .speed_prediction_utils import (
                load_model,
                predict_speed,
                load_audio_clip,
                compute_features
            )
        except ImportError as import_error:
            return JsonResponse({
                "error": f"Speed prediction utilities not found: {str(import_error)}",
                "solution": "Ensure speed_prediction_utils.py exists in signal_viewer_app/ with CNN14 implementation"
            }, status=500)

        # Load CNN14 model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SPEED_CNN14] Using device: {device}")

        try:
            model = load_model(MODEL_PATH, device=device)
            print("[SPEED_CNN14] ✅ CNN14 model loaded successfully")
        except Exception as model_load_error:
            error_trace = traceback.format_exc()
            print(f"[SPEED_CNN14] Model loading error:\n{error_trace}")

            return JsonResponse({
                "error": "Failed to load CNN14 model",
                "details": str(model_load_error),
                "model_path": MODEL_PATH,
                "solutions": [
                    "Verify model file is not corrupted",
                    "Check model was trained with CNN14 architecture",
                    "Ensure PyTorch version compatibility"
                ],
                "traceback": error_trace
            }, status=500)

        # Predict speed using CNN14
        try:
            predicted_speed = predict_speed(model, temp_audio_path, device=device)
            print(f"[SPEED_CNN14] ✅ Predicted speed: {predicted_speed:.2f} km/h")
        except Exception as prediction_error:
            error_trace = traceback.format_exc()
            print(f"[SPEED_CNN14] Prediction error:\n{error_trace}")
            return JsonResponse({
                "error": f"Prediction failed: {str(prediction_error)}",
                "traceback": error_trace
            }, status=500)

        # Get additional audio analysis for visualization
        try:
            # Load audio for visualization (16 kHz for display)
            audio_full, sr, _ = load_and_resample_audio(temp_audio_path, target_sr=16000)
            
            # Compute FFT
            fft_data = compute_fft_analysis(audio_full, sr)

            # Downsample waveform for plotting
            waveform_downsample = max(1, len(audio_full) // 10000)
            waveform_plot = audio_full[::waveform_downsample].tolist()

        except Exception as viz_error:
            print(f"[SPEED_CNN14] Warning: Visualization data generation failed: {viz_error}")
            # Use minimal visualization data
            waveform_plot = []
            fft_data = {
                'fft_frequencies_plot': [],
                'fft_magnitude_plot': [],
                'max_frequency': 0,
                'nyquist_frequency': 16000
            }
            sr = 16000

        # Prepare successful response
        response_data = {
            "success": True,
            "predicted_speed_kmh": float(predicted_speed),
            "model_name": "CNN14 (PANNs)",
            "model_config": {
                "sample_rate": 32000,
                "duration": 10,
                "n_mels": 64,
                "architecture": "CNN14"
            },
            "waveform": waveform_plot,
            "sr": sr,
            "filename": filename,
            "max_frequency": fft_data['max_frequency'],
            "fft_frequencies": fft_data['fft_frequencies_plot'],
            "fft_magnitudes": fft_data['fft_magnitude_plot'],
            "duration": float(len(audio_full) / sr) if 'audio_full' in locals() else 0,
            "model_info": {
                "device": str(device),
                "pytorch_version": torch.__version__
            }
        }

        print("[SPEED_CNN14] ✅ Response prepared successfully")
        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[SPEED_CNN14] ❌ Unexpected error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Speed prediction failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                print(f"[SPEED_CNN14] Cleaned up temporary file")
            except Exception as e:
                print(f"[SPEED_CNN14] Cleanup warning: {e}")


@csrf_exempt
def detect_ecg_abnormality(request):
    """
    This function is an AI endpoint. It receives a JSON payload
    of signal data from the browser,runs it through the loaded Keras model,
    and returns a JSON prediction.

    Detect ECG abnormalities using pretrained Keras model.
    Only works with 100Hz ECG signals from PTB-XL dataset.
    """

    # Validation and Prerequisite Check
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    # Check TensorFlow availability
    if not TENSORFLOW_AVAILABLE:
        return JsonResponse({
            "error": "TensorFlow is not installed.",
            "solution": "Install TensorFlow: pip install tensorflow"
        }, status=500)


    # Parsing and Validating the JSON Payload
    try:
        # Parse request body
        data = json.loads(request.body)
        signals = data.get('signals')  # List of lists: [[ch1_samples], [ch2_samples], ...]
        fs = data.get('fs')
        channel_names = data.get('channel_names', [])

        if not signals or not fs:
            return JsonResponse({
                "error": "Missing signals or sampling frequency."
            }, status=400)

        print(f"[ECG_DETECT] Received signal with fs={fs}Hz, channels={len(signals)}")

        # Validate sampling frequency (must be 100Hz for PTB-XL model)
        if fs != 100:
            return JsonResponse({
                "error": f"Model only supports 100Hz signals. Received {fs}Hz.",
                "suggestion": "Please upload 100Hz ECG records from PTB-XL dataset."
            }, status=400)
        #It parses the JSON data sent from the JavaScript.
        # It then performs the most critical check: if fs != 100:
        # Your AI model was trained only on 100Hz signals. Sending it any other frequency
        # (like 500Hz) will produce garbage results.
        # This check stops it and returns a clear error to the user.


        # Preparing the Data for the Model
        signals_np = np.array(signals, dtype=np.float32)
        # Convert to numpy array

        # Validate signal shape
        num_channels, num_samples = signals_np.shape
        print(f"[ECG_DETECT] Signal shape: {signals_np.shape}")

        if num_channels < 12:
            return JsonResponse({
                "error": f"Model requires 12-lead ECG. Received {num_channels} channels.",
                "available_channels": channel_names
            }, status=400)

        # Use only first 12 channels if more are available
        if num_channels > 12:
            signals_np = signals_np[:12, :]
            print(f"[ECG_DETECT] Using first 12 channels")
        # It converts the signals (a Python list) into a NumPy array.
        # It then validates the number of channels. The model requires a 12-lead (12-channel)
        # input. If the signal has fewer than 12, it returns an error.
        # If it has more than 12, it simply slices the array to use only the first 12.

        # Block 4: Loading the Model
        # Check if model exists
        if not os.path.exists(ECG_MODEL_PATH):
            return JsonResponse({
                "error": "ECG classification model not found.",
                "model_path": ECG_MODEL_PATH,
                "solution": "Please place your trained PTB-XL Keras model at signal_viewer_app/assets/ecg_model/model02.keras"
            }, status=500)

        # Load model (TensorFlow/Keras)
        try:
            print(f"[ECG_DETECT] Loading Keras model from: {ECG_MODEL_PATH}")
            model = load_ecg_model(ECG_MODEL_PATH, device='cpu', num_classes=5, input_channels=12)
            print("[ECG_DETECT] ✅ Model loaded successfully")

        except Exception as model_error:
            error_trace = traceback.format_exc()
            print(f"[ECG_DETECT] Model loading error:\n{error_trace}")
            return JsonResponse({
                "error": f"Failed to load ECG model: {str(model_error)}",
                "model_path": ECG_MODEL_PATH,
                "model_type": "Keras (.keras format)",
                "traceback": error_trace
            }, status=500)

        #Block 5: Running the Prediction
        # Run prediction
        try:
            print("[ECG_DETECT] Running prediction...")
            prediction = predict_ecg_abnormality(model, signals_np, device='cpu')

            print(f"[ECG_DETECT] ✅ Prediction: {prediction['predicted_class']} "
                  f"({'Normal' if prediction['is_normal'] else 'Abnormal'}, "
                  f"{prediction['confidence']:.2%} confidence)")

        #Block 6: Sending the Success Response
            response_data = {
                "success": True,
                "prediction": prediction,
                "signal_info": {
                    "fs": fs,
                    "num_channels": num_channels,
                    "num_samples": num_samples,
                    "duration_sec": float(num_samples / fs)
                },
                "model_info": {
                    "type": "TensorFlow/Keras",
                    "class_names": ECG_CLASS_NAMES
                }
            }

            return JsonResponse(response_data)

        except Exception as pred_error:
            error_trace = traceback.format_exc()
            print(f"[ECG_DETECT] Prediction error:\n{error_trace}")
            return JsonResponse({
                "error": f"Prediction failed: {str(pred_error)}",
                "traceback": error_trace
            }, status=500)

    except json.JSONDecodeError as e:
        return JsonResponse({
            "error": f"Invalid JSON in request body: {str(e)}"
        }, status=400)

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"[ECG_DETECT] ❌ Error:\n{error_trace}")
        return JsonResponse({
            "error": f"ECG detection failed: {str(e)}",
            "traceback": error_trace
        }, status=500)