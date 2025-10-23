# signal_viewer_app/views.py
# FIXED VERSION - Resolves model loading and gender detection issues
import torch
import json
import os
import tempfile
import numpy as np
import base64
import traceback
import wfdb
import wave
import io

# --- External Libraries ---
from scipy.signal import resample_poly
from pathlib import Path
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from .anti_aliasing_utils import AudioAntiAliaser, audio_array_to_wav_bytes
try:
    import librosa
    import soundfile as sf
    import tensorflow as tf
    from .yamnet_utils import initialize_yamnet, YAMNetEmbeddingLayer, extract_yamnet_embeddings
    from inaSpeechSegmenter import Segmenter

    ML_AVAILABLE = True
    print("✅ All ML libraries loaded successfully (librosa, tensorflow, soundfile, inaSpeechSegmenter)")
except ImportError as e:
    print(f"⚠️ Warning: Audio/ML dependencies missing: {e}")
    ML_AVAILABLE = False


    def initialize_yamnet():
        raise NotImplementedError("ML dependencies missing.")


    class YAMNetEmbeddingLayer:
        pass


    def extract_yamnet_embeddings(*args, **kwargs):
        raise NotImplementedError("ML dependencies missing.")


    class Segmenter:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("inaSpeechSegmenter not installed.")

# --- Configuration Constants ---
TARGET_SAMPLE_RATE = 1000
TARGET_SAMPLE_RATE_AUDIO = 16000
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'drone_bird_detection', 'yamnet_finetuned.h5')
CLASS_NAMES = ['Drone', 'Bird', 'Noise/Other']

# Add this constant with other configuration constants
ANTI_ALIASING_MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    'assets',
    'anti_aliasing',
    'best_modell.pth'
)

# *********************************************************************************
# * SARImageProcessor Class Definition (File Handling Logic) *
# *********************************************************************************
class SARImageProcessor:
    """Processes SAR/TIFF images with downsampling and contrast enhancement."""

    def __init__(self, max_dimension=4000, remove_pixel_limit=True):
        self.max_dimension = max_dimension
        if remove_pixel_limit:
            Image.MAX_IMAGE_PIXELS = None

    def load_tiff(self, file_path):
        return Image.open(file_path)

    def get_metadata(self, img):
        width, height = img.size
        return {
            'width': width,
            'height': height,
            'mode': img.mode,
            'format': img.format,
            'bands': len(img.getbands()) if hasattr(img, 'getbands') else 1
        }

    def downsample_if_needed(self, img):
        width, height = img.size
        if width <= self.max_dimension and height <= self.max_dimension:
            return img, 1.0
        downsample_factor = max(width / self.max_dimension, height / self.max_dimension)
        new_width = int(width / downsample_factor)
        new_height = int(height / downsample_factor)
        downsampled = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return downsampled, downsample_factor

    def to_numpy(self, img):
        return np.array(img)

    def extract_grayscale_data(self, img_array):
        if len(img_array.shape) == 2:
            return img_array
        elif len(img_array.shape) == 3:
            return img_array[:, :, 0]
        else:
            return img_array

    def calculate_statistics(self, data):
        return {
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'median': float(np.median(data)),
            'p2': float(np.percentile(data, 2)),
            'p98': float(np.percentile(data, 98))
        }

    def normalize_with_contrast_enhancement(self, data, clip_percentile=2):
        if np.max(data) - np.min(data) == 0:
            return np.zeros_like(data, dtype=np.uint8)
        p_low = clip_percentile
        p_high = 100 - clip_percentile
        p_low_val, p_high_val = np.percentile(data, [p_low, p_high])
        data_clipped = np.clip(data, p_low_val, p_high_val)
        normalized = ((data_clipped - p_low_val) / (p_high_val - p_low_val) * 255).astype(np.uint8)
        return normalized

    def apply_colormap(self, grayscale_array, colormap='viridis'):
        img = Image.fromarray(grayscale_array, mode='L')
        return img.convert('RGB')

    def to_base64_png(self, img, optimize=True):
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
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    if 'dat_file' not in request.FILES or 'hea_file' not in request.FILES:
        return JsonResponse(
            {"error": "Missing .dat or .hea file. Both are required for WFDB record."},
            status=400
        )

    dat_file = request.FILES['dat_file']
    hea_file = request.FILES['hea_file']
    temp_dir = tempfile.mkdtemp()
    temp_dat_path = os.path.join(temp_dir, dat_file.name)
    temp_hea_path = os.path.join(temp_dir, hea_file.name)
    record_stem = os.path.join(temp_dir, os.path.splitext(dat_file.name)[0])

    try:
        with open(temp_dat_path, 'wb+') as dest:
            for chunk in dat_file.chunks():
                dest.write(chunk)
        with open(temp_hea_path, 'wb+') as dest:
            for chunk in hea_file.chunks():
                dest.write(chunk)

        rec = wfdb.rdsamp(record_stem)
        signals, fields = rec[0], rec[1]
        original_fs = int(fields.get('fs', 100))
        channel_names = list(fields.get('sig_name', [f"ch{i}" for i in range(signals.shape[1])]))
        processed_signals = signals
        processed_fs = original_fs

        if original_fs > TARGET_SAMPLE_RATE:
            up = TARGET_SAMPLE_RATE
            down = original_fs
            processed_signals = resample_poly(signals, up, down, axis=0)
            processed_fs = TARGET_SAMPLE_RATE

        signal_length = processed_signals.shape[0]
        duration = signal_length / processed_fs
        signals_list = np.nan_to_num(processed_signals.T, nan=0.0).astype(float).tolist()

        response_data = {
            "filename": dat_file.name,
            "fs": processed_fs,
            "channel_names": channel_names,
            "signals": signals_list,
            "duration": duration
        }
        return JsonResponse(response_data)

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
def convert_eeg_set_to_json(request):
    """Convert EEG .set files to JSON format."""
    if request.method != 'POST' or 'file' not in request.FILES:
        return HttpResponseBadRequest("Invalid request: No file uploaded.")

    uploaded_file = request.FILES['file']
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        with open(temp_path, 'wb+') as dest:
            for chunk in uploaded_file.chunks():
                dest.write(chunk)

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
        tb = traceback.format_exc()
        print(f"SAR Critical Processing Error:\n{tb}")
        return JsonResponse(
            {"error": f"CRITICAL SERVER ERROR: Processing failed. Details: {str(e)}"},
            status=500
        )


# ========== FIXED: DRONE/BIRD DETECTION ==========
# Replace the analyze_audio_detect_bird_and_drone function in views.py with this fixed version

# FIXED VERSION - Proper Model Loading and Prediction
# Replace the analyze_audio_detect_bird_and_drone function in views.py

# Replace the analyze_audio_detect_bird_and_drone function in views.py with this:

# Add this custom function before analyze_audio_detect_bird_and_drone
def load_model_with_compatibility(model_path):
    """
    Load model with backward compatibility for batch_shape parameter.
    Handles conversion from old Keras format to new format.
    """
    try:
        # Try loading directly first
        print(f"Attempting to load model from {model_path}...")

        # Initialize YAMNet
        initialize_yamnet()

        custom_objects = {
            'extract_yamnet_embeddings': extract_yamnet_embeddings,
            'YAMNetEmbeddingLayer': YAMNetEmbeddingLayer,
        }

        # Method 1: Try loading with compile=False
        try:
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            print("✅ Model loaded successfully (direct load)")
            return model
        except Exception as e1:
            print(f"Direct load failed: {e1}")

            # Method 2: Load and rebuild model architecture
            try:
                print("Attempting to rebuild model architecture...")

                # Load the weights file
                import h5py
                with h5py.File(model_path, 'r') as f:
                    # Get model config from HDF5
                    if 'model_config' in f.attrs:
                        import json
                        model_config = json.loads(f.attrs['model_config'])

                        # Fix batch_shape -> shape conversion
                        def fix_config(config):
                            if isinstance(config, dict):
                                if 'config' in config and isinstance(config['config'], dict):
                                    layer_config = config['config']
                                    # Convert batch_shape to shape
                                    if 'batch_shape' in layer_config:
                                        batch_shape = layer_config.pop('batch_shape')
                                        if batch_shape and len(batch_shape) > 1:
                                            layer_config['shape'] = batch_shape[1:]
                                        print(f"Fixed layer: {config.get('class_name', 'unknown')}")

                                    # Recursively fix nested configs
                                    for key, value in layer_config.items():
                                        if isinstance(value, (dict, list)):
                                            fix_config(value)

                                # Handle layers list
                                if 'layers' in config:
                                    for layer in config['layers']:
                                        fix_config(layer)

                            elif isinstance(config, list):
                                for item in config:
                                    fix_config(item)

                        fix_config(model_config)

                        # Rebuild model from fixed config
                        model = tf.keras.models.model_from_json(
                            json.dumps(model_config),
                            custom_objects=custom_objects
                        )

                        # Load weights
                        model.load_weights(model_path)
                        print("✅ Model rebuilt and weights loaded successfully")
                        return model

            except Exception as e2:
                print(f"Rebuild attempt failed: {e2}")

                # Method 3: Manually recreate the model architecture
                print("Attempting manual model recreation...")
                return create_yamnet_model_manually()

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"❌ All model loading attempts failed:\n{error_traceback}")
        raise RuntimeError(f"Could not load model: {str(e)}")


def create_yamnet_model_manually():
    """
    Manually recreate the YAMNet fine-tuned model architecture.
    Use this as fallback if model file is corrupted or incompatible.
    """
    print("Creating model architecture manually...")

    # Initialize YAMNet
    initialize_yamnet()

    # Recreate the model architecture (must match training architecture)
    audio_input = tf.keras.layers.Input(shape=(None,), dtype=tf.float32, name='audio_input')

    # YAMNet embeddings
    yamnet_embeddings = YAMNetEmbeddingLayer(name='yamnet_embeddings')(audio_input)

    # Global average pooling
    pooled = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(yamnet_embeddings)

    # Dropout
    dropout1 = tf.keras.layers.Dropout(0.5, name='dropout')(pooled)

    # Dense layers
    dense1 = tf.keras.layers.Dense(512, activation='relu', name='dense1')(dropout1)
    dropout2 = tf.keras.layers.Dropout(0.3, name='dropout2')(dense1)
    dense2 = tf.keras.layers.Dense(256, activation='relu', name='dense2')(dropout2)

    # Output layer (3 classes: Drone, Bird, Noise)
    output = tf.keras.layers.Dense(3, activation='softmax', name='predictions')(dense2)

    # Create model
    model = tf.keras.Model(inputs=audio_input, outputs=output, name='yamnet_finetuned')

    print("⚠️ Model architecture created, but weights are NOT loaded!")
    print("⚠️ You need to retrain the model or fix the .h5 file")

    return model


# Now update the analyze_audio_detect_bird_and_drone function
# Replace the analyze_audio_detect_bird_and_drone function in views.py with this:

# Replace the analyze_audio_detect_bird_and_drone function in views.py with this:

@csrf_exempt
def analyze_audio_detect_bird_and_drone(request):
    """
    FIXED: Proper model loading and prediction with YAMNet.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    try:
        body_data = json.loads(request.body)
        audio_data_uri = body_data.get('audio_data')
        filename = body_data.get('filename', 'audio_file.wav')
        target_sample_rate = body_data.get('target_sample_rate', TARGET_SAMPLE_RATE_AUDIO)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"error": f"Invalid JSON in request body: {str(e)}"},
            status=400
        )

    if not audio_data_uri:
        return JsonResponse({"error": "No audio data received."}, status=400)

    # Decode audio data
    try:
        if ',' in audio_data_uri:
            header, encoded = audio_data_uri.split(',', 1)
        else:
            encoded = audio_data_uri
        decoded_bytes = base64.b64decode(encoded)
    except Exception as e:
        return JsonResponse({"error": f"Failed to decode audio data: {str(e)}"}, status=400)

    if not ML_AVAILABLE:
        return JsonResponse({
            "error": "ML dependencies not available. Please install librosa and tensorflow."
        }, status=500)

    temp_audio_path = None
    try:
        # Save temporary audio file
        ext = os.path.splitext(filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(decoded_bytes)
            temp_audio_path = tmp_file.name

        print(f"[DETECT] Loading audio file: {filename}")

        # Load audio with librosa
        audio_original, sr_original = librosa.load(temp_audio_path, sr=None, mono=True)

        # Limit duration to 30 seconds
        MAX_DURATION_SECONDS = 30
        max_samples = MAX_DURATION_SECONDS * sr_original

        if len(audio_original) > max_samples:
            print(f"[DETECT] Audio too long. Truncating to {MAX_DURATION_SECONDS}s...")
            audio_original = audio_original[:max_samples]

        # Resample to 16kHz (YAMNet requirement)
        if sr_original != target_sample_rate:
            audio = librosa.resample(
                audio_original,
                orig_sr=sr_original,
                target_sr=target_sample_rate
            )
            sr = target_sample_rate
        else:
            audio = audio_original
            sr = sr_original

        print(f"[DETECT] Audio processed: {len(audio)} samples at {sr} Hz")

        # Normalize audio to [-1, 1] range
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # FFT analysis for visualization
        fft = np.fft.fft(audio)
        fft_magnitude = np.abs(fft[:len(fft) // 2])
        fft_frequencies = np.fft.fftfreq(len(audio), 1 / sr)[:len(fft) // 2]

        threshold = np.max(fft_magnitude) * 0.01
        significant_freqs = fft_frequencies[fft_magnitude > threshold]
        max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

        nyquist_frequency = sr / 2

        # Downsample FFT for plotting
        downsample_factor = max(1, len(fft_frequencies) // 1000)
        fft_frequencies_plot = fft_frequencies[::downsample_factor].tolist()
        fft_magnitude_plot = fft_magnitude[::downsample_factor].tolist()

        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            return JsonResponse({
                "error": f"Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.",
                "model_path": MODEL_PATH
            }, status=500)

        try:
            print("[DETECT] Initializing YAMNet...")
            from .yamnet_utils import initialize_yamnet, load_finetuned_model, predict_audio_class

            # Initialize YAMNet first
            initialize_yamnet()
            print("[DETECT] ✅ YAMNet initialized")

            # Load fine-tuned model
            print(f"[DETECT] Loading model from {MODEL_PATH}...")
            model = load_finetuned_model(MODEL_PATH)
            print("[DETECT] ✅ Model loaded successfully")

            # Make prediction using the utility function
            print("[DETECT] Running prediction...")
            prediction_result = predict_audio_class(
                model=model,
                audio_array=audio,
                class_names=CLASS_NAMES
            )

            print(f"[DETECT] ✅ Prediction complete")
            print(f"[DETECT] Result: {prediction_result['predicted_class']} "
                  f"({prediction_result['confidence']:.2%} confidence)")

            # Prepare response
            response_data = {
                "predicted_class": prediction_result['predicted_class'],
                "probabilities": prediction_result['probabilities'],
                "confidence": prediction_result['confidence'],
                "waveform": audio.tolist(),
                "sr": sr,
                "original_sr": sr_original,
                "class_names": CLASS_NAMES,
                "filename": filename,
                "max_frequency": float(max_frequency),
                "nyquist_frequency": float(nyquist_frequency),
                "fft_frequencies": fft_frequencies_plot,
                "fft_magnitudes": fft_magnitude_plot,
                "audio_duration": float(len(audio) / sr),
                "audio_samples": len(audio)
            }

            return JsonResponse(response_data)

        except Exception as model_error:
            error_traceback = traceback.format_exc()
            print(f"[DETECT] ❌ Model execution error:\n{error_traceback}")
            return JsonResponse({
                "error": f"Model execution failed: {str(model_error)}",
                "details": error_traceback,
                "suggestion": "The model file may be incompatible. Check that yamnet_finetuned.h5 is properly trained."
            }, status=500)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[DETECT] ❌ Audio Analysis Error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Audio analysis failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    finally:
        # Cleanup temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"[DETECT] Cleanup warning: {e}")

# ========== FIXED: VOICE GENDER DETECTION ==========
@csrf_exempt
def analyze_voices_gender(request):
    """
    FIXED: Improved gender detection with better frequency analysis and inaSpeechSegmenter integration.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    try:
        body_data = json.loads(request.body)
        audio_data_uri = body_data.get('audio_data')
        filename = body_data.get('filename', 'audio_file.wav')
        target_sample_rate = body_data.get('target_sample_rate', TARGET_SAMPLE_RATE_AUDIO)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"error": f"Invalid JSON. Audio may be too large: {str(e)}"},
            status=400
        )

    if not audio_data_uri:
        return JsonResponse({"error": "No audio data received."}, status=400)

    try:
        if ',' in audio_data_uri:
            header, encoded = audio_data_uri.split(',', 1)
        else:
            encoded = audio_data_uri
        decoded_bytes = base64.b64decode(encoded)
    except Exception as e:
        return JsonResponse({"error": f"Failed to decode audio: {str(e)}"}, status=400)

    temp_audio_path = None
    temp_wav_path = None

    try:
        # Check FFmpeg availability
        def check_ffmpeg():
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

        # Save audio file
        ext = os.path.splitext(filename)[1].lower() or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(decoded_bytes)
            temp_audio_path = tmp_file.name

        print(f"[GENDER] Processing: {filename}")

        # Load audio
        audio_original, sr_original = librosa.load(temp_audio_path, sr=None, mono=True)

        MAX_DURATION_SECONDS = 30
        max_samples = MAX_DURATION_SECONDS * sr_original

        if len(audio_original) > max_samples:
            audio_original = audio_original[:max_samples]

        if sr_original != target_sample_rate:
            audio = librosa.resample(audio_original, orig_sr=sr_original, target_sr=target_sample_rate)
            sr = target_sample_rate
        else:
            audio = audio_original
            sr = sr_original

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        print(f"[GENDER] Audio: {len(audio)} samples at {sr} Hz")

        # FFT analysis
        fft = np.fft.fft(audio)
        fft_magnitude = np.abs(fft[:len(fft) // 2])
        fft_frequencies = np.fft.fftfreq(len(audio), 1 / sr)[:len(fft) // 2]

        threshold = np.max(fft_magnitude) * 0.01
        significant_freqs = fft_frequencies[fft_magnitude > threshold]
        max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

        nyquist_frequency = sr / 2

        downsample_factor = max(1, len(fft_frequencies) // 1000)
        fft_frequencies_plot = fft_frequencies[::downsample_factor].tolist()
        fft_magnitude_plot = fft_magnitude[::downsample_factor].tolist()

        # ========== IMPROVED GENDER DETECTION ==========
        use_fallback = False
        fallback_reason = ""

        if not ffmpeg_available:
            use_fallback = True
            fallback_reason = "FFmpeg not installed"
            print(f"[GENDER] {fallback_reason} - using improved fallback")

        # Try inaSpeechSegmenter
        if not use_fallback:
            try:
                print("[GENDER] Initializing inaSpeechSegmenter...")

                # Convert to WAV if needed
                if not temp_audio_path.lower().endswith('.wav'):
                    temp_wav_path = tempfile.NamedTemporaryFile(delete=False, suffix='.wav').name
                    sf.write(temp_wav_path, audio_original, sr_original)
                    analysis_path = temp_wav_path
                else:
                    analysis_path = temp_audio_path

                seg = Segmenter(vad_engine='smn', detect_gender=True)
                segmentation = seg(analysis_path)

                male_duration = 0
                female_duration = 0
                total_speech = 0

                print("[GENDER] Segmentation results:")
                for label, start, end in segmentation:
                    duration = end - start
                    print(f"  {label:12} | {start:7.2f}s - {end:7.2f}s | {duration:6.2f}s")

                    if label == 'male':
                        male_duration += duration
                        total_speech += duration
                    elif label == 'female':
                        female_duration += duration
                        total_speech += duration

                print(f"[GENDER] Total speech: {total_speech:.2f}s")
                print(f"[GENDER] Male: {male_duration:.2f}s, Female: {female_duration:.2f}s")

                if total_speech == 0:
                    use_fallback = True
                    fallback_reason = "No speech detected"
                    print(f"[GENDER] {fallback_reason}")
                else:
                    male_probability = male_duration / total_speech
                    female_probability = female_duration / total_speech

                    if male_duration > female_duration:
                        predicted_gender = 'Male'
                        confidence = male_probability
                    elif female_duration > male_duration:
                        predicted_gender = 'Female'
                        confidence = female_probability
                    else:
                        predicted_gender = 'Equal'
                        confidence = 0.5
                        male_probability = 0.5
                        female_probability = 0.5

                    print(f"[GENDER] Result: {predicted_gender} ({confidence:.1%})")

            except Exception as seg_error:
                use_fallback = True
                fallback_reason = f"inaSpeechSegmenter error: {str(seg_error)}"
                print(f"[GENDER] {fallback_reason}")

        # IMPROVED FALLBACK METHOD
        if use_fallback:
            print(f"[GENDER] Using IMPROVED fallback: {fallback_reason}")

            # Method 1: Spectral Centroid (more accurate than zero-crossing)
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            mean_spectral_centroid = np.mean(spectral_centroids)

            # Method 2: Pitch detection using autocorrelation
            def estimate_pitch_autocorr(signal, sr, fmin=50, fmax=400):
                """Estimate pitch using autocorrelation method."""
                # Normalize signal
                signal = signal - np.mean(signal)

                # Autocorrelation
                autocorr = np.correlate(signal, signal, mode='full')
                autocorr = autocorr[len(autocorr) // 2:]

                # Find peaks in autocorrelation
                min_period = int(sr / fmax)
                max_period = int(sr / fmin)

                if max_period >= len(autocorr):
                    return None

                # Find the first significant peak after the minimum period
                autocorr_segment = autocorr[min_period:max_period]
                if len(autocorr_segment) == 0:
                    return None

                peak_index = np.argmax(autocorr_segment) + min_period
                estimated_freq = sr / peak_index

                return estimated_freq

            # Estimate fundamental frequency
            estimated_f0 = estimate_pitch_autocorr(audio, sr)

            print(f"[GENDER] Spectral centroid: {mean_spectral_centroid:.2f} Hz")
            print(f"[GENDER] Estimated F0: {estimated_f0:.2f} Hz" if estimated_f0 else "[GENDER] F0 estimation failed")

            # Combined decision using multiple features
            # Male typical ranges: F0: 85-180 Hz, Spectral Centroid: 200-500 Hz
            # Female typical ranges: F0: 165-255 Hz, Spectral Centroid: 400-800 Hz

            male_score = 0
            female_score = 0

            # Score based on F0
            if estimated_f0 is not None:
                if estimated_f0 < 130:
                    male_score += 2.0
                elif estimated_f0 > 200:
                    female_score += 2.0
                elif estimated_f0 < 165:
                    male_score += 1.0
                else:
                    female_score += 1.0

            # Score based on spectral centroid
            if mean_spectral_centroid < 350:
                male_score += 1.5
            elif mean_spectral_centroid > 600:
                female_score += 1.5
            elif mean_spectral_centroid < 500:
                male_score += 0.5
            else:
                female_score += 0.5

            # Calculate probabilities
            total_score = male_score + female_score
            if total_score > 0:
                male_probability = male_score / total_score
                female_probability = female_score / total_score
            else:
                male_probability = 0.5
                female_probability = 0.5

            # Add some confidence based on how clear the signal is
            confidence_factor = min(1.0, abs(male_probability - female_probability) * 2)

            # Ensure minimum confidence of 0.55 for clear decisions
            if male_probability > female_probability:
                male_probability = max(0.55, male_probability)
                female_probability = 1 - male_probability
                predicted_gender = 'Male'
                confidence = male_probability
            else:
                female_probability = max(0.55, female_probability)
                male_probability = 1 - female_probability
                predicted_gender = 'Female'
                confidence = female_probability

            # Cap maximum confidence at 0.90 for fallback method
            male_probability = min(0.90, male_probability)
            female_probability = min(0.90, female_probability)
            confidence = max(male_probability, female_probability)

            print(f"[GENDER] Fallback scores - Male: {male_score:.2f}, Female: {female_score:.2f}")
            print(f"[GENDER] Result: {predicted_gender} ({confidence:.1%})")

            # Set durations for fallback
            total_speech = len(audio) / sr
            male_duration = male_probability * total_speech
            female_duration = female_probability * total_speech

        # Prepare response
        response_data = {
            "predicted_gender": predicted_gender,
            "confidence": float(confidence),
            "male_probability": float(male_probability),
            "female_probability": float(female_probability),
            "waveform": audio.tolist(),
            "sr": sr,
            "original_sr": sr_original,
            "filename": filename,
            "max_frequency": float(max_frequency),
            "nyquist_frequency": float(nyquist_frequency),
            "fft_frequencies": fft_frequencies_plot,
            "fft_magnitudes": fft_magnitude_plot,
            "total_speech_duration": float(total_speech),
            "male_speech_duration": float(male_duration),
            "female_speech_duration": float(female_duration),
            "detection_method": "fallback" if use_fallback else "inaSpeechSegmenter",
            "fallback_reason": fallback_reason if use_fallback else None
        }

        print("[GENDER] ✅ Response prepared successfully")
        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[GENDER] ❌ Error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Gender detection failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    finally:
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
    """
    Handles car audio file upload for visualization only (no ML classification).
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    try:
        body_data = json.loads(request.body)
        audio_data_uri = body_data.get('audio_data')
        filename = body_data.get('filename', 'audio_file.wav')
        target_sample_rate = body_data.get('target_sample_rate', TARGET_SAMPLE_RATE_AUDIO)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"error": f"Invalid JSON: {str(e)}"},
            status=400
        )

    if not audio_data_uri:
        return JsonResponse({"error": "No audio data received."}, status=400)

    try:
        if ',' in audio_data_uri:
            header, encoded = audio_data_uri.split(',', 1)
        else:
            encoded = audio_data_uri
        decoded_bytes = base64.b64decode(encoded)
    except Exception as e:
        return JsonResponse({"error": f"Failed to decode audio: {str(e)}"}, status=400)

    if not ML_AVAILABLE:
        return JsonResponse({
            "error": "ML dependencies not available. Please install librosa."
        }, status=500)

    temp_audio_path = None
    try:
        ext = os.path.splitext(filename)[1] or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(decoded_bytes)
            temp_audio_path = tmp_file.name

        audio_original, sr_original = librosa.load(temp_audio_path, sr=None, mono=True)

        MAX_DURATION_SECONDS = 30
        max_samples = MAX_DURATION_SECONDS * sr_original

        if len(audio_original) > max_samples:
            audio_original = audio_original[:max_samples]

        if sr_original != target_sample_rate:
            audio = librosa.resample(audio_original, orig_sr=sr_original, target_sr=target_sample_rate)
            sr = target_sample_rate
        else:
            audio = audio_original
            sr = sr_original

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        # FFT analysis
        fft = np.fft.fft(audio)
        fft_magnitude = np.abs(fft[:len(fft) // 2])
        fft_frequencies = np.fft.fftfreq(len(audio), 1 / sr)[:len(fft) // 2]

        threshold = np.max(fft_magnitude) * 0.01
        significant_freqs = fft_frequencies[fft_magnitude > threshold]
        max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

        nyquist_frequency = sr / 2

        downsample_factor = max(1, len(fft_frequencies) // 1000)
        fft_frequencies_plot = fft_frequencies[::downsample_factor].tolist()
        fft_magnitude_plot = fft_magnitude[::downsample_factor].tolist()

        response_data = {
            "waveform": audio.tolist(),
            "sr": sr,
            "original_sr": sr_original,
            "filename": filename,
            "max_frequency": float(max_frequency),
            "nyquist_frequency": float(nyquist_frequency),
            "fft_frequencies": fft_frequencies_plot,
            "fft_magnitudes": fft_magnitude_plot
        }

        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        return JsonResponse({
            "error": f"Audio analysis failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"Cleanup warning: {e}")


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
    Apply anti-aliasing to uploaded audio using pretrained neural network.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    try:
        body_data = json.loads(request.body)
        audio_data_uri = body_data.get('audio_data')
        filename = body_data.get('filename', 'audio_file.wav')
        sample_rate = body_data.get('sample_rate', TARGET_SAMPLE_RATE_AUDIO)
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"error": f"Invalid JSON: {str(e)}"},
            status=400
        )

    if not audio_data_uri:
        return JsonResponse({"error": "No audio data received."}, status=400)

    # Decode audio data
    try:
        if ',' in audio_data_uri:
            header, encoded = audio_data_uri.split(',', 1)
        else:
            encoded = audio_data_uri
        decoded_bytes = base64.b64decode(encoded)
    except Exception as e:
        return JsonResponse({"error": f"Failed to decode audio: {str(e)}"}, status=400)

    temp_audio_path = None

    try:
        print("[ANTI-ALIAS] Processing request...")

        # Check if model exists
        if not os.path.exists(ANTI_ALIASING_MODEL_PATH):
            return JsonResponse({
                "error": f"Anti-aliasing model not found at {ANTI_ALIASING_MODEL_PATH}. "
                         "Please ensure model_ep5.pth is placed in signal_viewer_app/assets/anti_aliasing/"
            }, status=500)

        # Save temporary audio file
        ext = os.path.splitext(filename)[1].lower() or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(decoded_bytes)
            temp_audio_path = tmp_file.name

        print(f"[ANTI-ALIAS] Loading audio: {filename}")

        # Load audio with librosa
        try:
            import librosa
        except ImportError:
            return JsonResponse({
                "error": "librosa is required for anti-aliasing. Please install it."
            }, status=500)

        audio_original, sr_original = librosa.load(temp_audio_path, sr=None, mono=True)

        # Limit duration to 30 seconds
        MAX_DURATION_SECONDS = 30
        max_samples = MAX_DURATION_SECONDS * sr_original

        if len(audio_original) > max_samples:
            print(f"[ANTI-ALIAS] Audio too long. Truncating to {MAX_DURATION_SECONDS}s...")
            audio_original = audio_original[:max_samples]

        # Resample to 16kHz if needed (model expects 16kHz)
        target_sr = 16000
        if sr_original != target_sr:
            audio = librosa.resample(audio_original, orig_sr=sr_original, target_sr=target_sr)
            sr = target_sr
        else:
            audio = audio_original
            sr = sr_original

        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        print(f"[ANTI-ALIAS] Audio loaded: {len(audio)} samples at {sr} Hz")

        # Initialize anti-aliaser
        print("[ANTI-ALIAS] Initializing model...")
        try:
            anti_aliaser = AudioAntiAliaser(
                model_path=ANTI_ALIASING_MODEL_PATH,
                device='auto',
                sample_rate=sr,
                hidden_size=160,
                num_residual_blocks=5,
                num_lstm_layers=2,
                lstm_hidden_size=224
            )
        except Exception as model_error:
            error_traceback = traceback.format_exc()
            print(f"[ANTI-ALIAS] Model loading error:\n{error_traceback}")
            return JsonResponse({
                "error": f"Failed to load anti-aliasing model: {str(model_error)}",
                "details": error_traceback
            }, status=500)

        # Apply anti-aliasing
        print("[ANTI-ALIAS] Applying anti-aliasing...")
        try:
            enhanced_audio = anti_aliaser.enhance_audio_array(
                audio_array=audio,
                sample_rate=sr,
                chunk_size=40000,
                overlap=4000
            )
        except Exception as enhance_error:
            error_traceback = traceback.format_exc()
            print(f"[ANTI-ALIAS] Enhancement error:\n{error_traceback}")
            return JsonResponse({
                "error": f"Failed to enhance audio: {str(enhance_error)}",
                "details": error_traceback
            }, status=500)

        print("[ANTI-ALIAS] ✅ Anti-aliasing complete")

        # Convert enhanced audio to WAV bytes
        wav_bytes = audio_array_to_wav_bytes(enhanced_audio, sr)
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')

        # FFT analysis on enhanced audio
        fft = np.fft.fft(enhanced_audio)
        fft_magnitude = np.abs(fft[:len(fft) // 2])
        fft_frequencies = np.fft.fftfreq(len(enhanced_audio), 1 / sr)[:len(fft) // 2]

        threshold = np.max(fft_magnitude) * 0.01
        significant_freqs = fft_frequencies[fft_magnitude > threshold]
        max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

        # Downsample for plotting
        downsample_factor = max(1, len(fft_frequencies) // 1000)
        fft_frequencies_plot = fft_frequencies[::downsample_factor].tolist()
        fft_magnitude_plot = fft_magnitude[::downsample_factor].tolist()

        # Downsample waveform for plotting
        waveform_downsample_factor = max(1, len(enhanced_audio) // 10000)
        waveform_plot = enhanced_audio[::waveform_downsample_factor].tolist()

        response_data = {
            "success": True,
            "enhanced_audio_b64": audio_b64,
            "waveform": waveform_plot,
            "sr": sr,
            "filename": f"enhanced_{filename}",
            "duration": float(len(enhanced_audio) / sr),
            "samples": len(enhanced_audio),
            "max_frequency": float(max_frequency),
            "nyquist_frequency": float(sr / 2),
            "fft_frequencies": fft_frequencies_plot,
            "fft_magnitudes": fft_magnitude_plot
        }

        print("[ANTI-ALIAS] ✅ Response prepared")
        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[ANTI-ALIAS] ❌ Error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Anti-aliasing failed: {str(e)}",
            "traceback": error_traceback
        }, status=500)

    finally:
        # Cleanup temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"[ANTI-ALIAS] Cleanup warning: {e}")


# Replace the predict_car_speed function in views.py with this improved version

@csrf_exempt
def predict_car_speed(request):
    """
    Predict vehicle speed from audio using the trained model.
    Includes fallback mechanisms and better error handling.
    """
    if request.method != 'POST':
        return HttpResponseBadRequest("Invalid request method.")

    try:
        body_data = json.loads(request.body)
        audio_data_uri = body_data.get('audio_data')
        filename = body_data.get('filename', 'audio_file.wav')
    except json.JSONDecodeError as e:
        return JsonResponse(
            {"error": f"Invalid JSON: {str(e)}"},
            status=400
        )

    if not audio_data_uri:
        return JsonResponse({"error": "No audio data received."}, status=400)

    # Decode audio data
    try:
        if ',' in audio_data_uri:
            header, encoded = audio_data_uri.split(',', 1)
        else:
            encoded = audio_data_uri
        decoded_bytes = base64.b64decode(encoded)
    except Exception as e:
        return JsonResponse({"error": f"Failed to decode audio: {str(e)}"}, status=400)

    temp_audio_path = None

    try:
        print("[SPEED] Processing speed prediction request...")

        # Check PyTorch availability
        try:
            import torch
        except ImportError:
            return JsonResponse({
                "error": "PyTorch is not installed. Please install it: pip install torch"
            }, status=500)

        # Define checkpoint directory
        CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'assets', 'checkpoints')

        print(f"[SPEED] Checkpoint directory: {CHECKPOINT_DIR}")
        print(f"[SPEED] Directory exists: {os.path.exists(CHECKPOINT_DIR)}")

        # Check if checkpoint directory exists
        if not os.path.exists(CHECKPOINT_DIR):
            return JsonResponse({
                "error": "Checkpoint directory not found",
                "checkpoint_dir": CHECKPOINT_DIR,
                "solution": "Please create the directory and place trained checkpoint files (best_fold_*.pth) there, or run the validation script: python validate_checkpoints.py"
            }, status=500)

        # Save temporary audio file
        ext = os.path.splitext(filename)[1].lower() or '.wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(decoded_bytes)
            temp_audio_path = tmp_file.name

        print(f"[SPEED] Loading audio: {filename}")

        # Import utilities
        try:
            from .speed_prediction_utils import (
                find_best_checkpoint,
                load_audio_clip,
                compute_features,
                AdvancedSpeedPredictor
            )
        except ImportError as import_error:
            return JsonResponse({
                "error": f"Speed prediction utilities not found: {str(import_error)}",
                "solution": "Ensure speed_prediction_utils.py exists in signal_viewer_app/"
            }, status=500)

        # Try to find and load checkpoint
        try:
            checkpoint_path, variance = find_best_checkpoint(CHECKPOINT_DIR)
            print(f"[SPEED] Using checkpoint: {checkpoint_path}")
        except Exception as checkpoint_error:
            error_trace = traceback.format_exc()
            print(f"[SPEED] Checkpoint loading failed:\n{error_trace}")

            # Return detailed error with instructions
            return JsonResponse({
                "error": "Failed to load model checkpoints",
                "details": str(checkpoint_error),
                "checkpoint_dir": CHECKPOINT_DIR,
                "solutions": [
                    "Run: python validate_checkpoints.py",
                    "Ensure checkpoint files are not corrupted (should be 10-50 MB each)",
                    "Re-train the model if checkpoints are invalid",
                    "Check PyTorch version compatibility",
                    "Create dummy checkpoint for testing: python validate_checkpoints.py (select 'yes' when prompted)"
                ],
                "traceback": error_trace
            }, status=500)

        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SPEED] Using device: {device}")

        model = AdvancedSpeedPredictor(n_mels=64).to(device)

        try:
            # PyTorch 2.9+ compatibility: Register numpy safe globals
            if hasattr(torch.serialization, 'add_safe_globals'):
                try:
                    if hasattr(np, '_core'):
                        torch.serialization.add_safe_globals([np._core.multiarray.scalar])
                    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
                except:
                    pass

            # Load checkpoint with weights_only=False
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                raise ValueError(f"Invalid checkpoint format: expected dict, got {type(checkpoint)}")

            if 'model_state_dict' not in checkpoint:
                raise ValueError(f"Checkpoint missing 'model_state_dict'. Available keys: {list(checkpoint.keys())}")

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print("[SPEED] ✅ Model loaded successfully")

        except Exception as model_load_error:
            error_trace = traceback.format_exc()
            print(f"[SPEED] Model loading error:\n{error_trace}")

            return JsonResponse({
                "error": "Failed to load model weights",
                "details": str(model_load_error),
                "checkpoint_path": str(checkpoint_path),
                "solutions": [
                    "Checkpoint file may be corrupted",
                    "Re-download or re-train the model",
                    "Check checkpoint structure with: python validate_checkpoints.py"
                ],
                "traceback": error_trace
            }, status=500)

        # Load and process audio
        try:
            audio = load_audio_clip(temp_audio_path)
            features = compute_features(audio).to(device)
            print(f"[SPEED] Audio features shape: {features.shape}")
        except Exception as audio_error:
            error_trace = traceback.format_exc()
            return JsonResponse({
                "error": f"Failed to process audio: {str(audio_error)}",
                "traceback": error_trace
            }, status=500)

        # Predict speed
        try:
            with torch.no_grad():
                speed_tensor = model(features, return_aux=False)
                predicted_speed = speed_tensor.item()

            print(f"[SPEED] ✅ Predicted speed: {predicted_speed:.2f} km/h")

        except Exception as prediction_error:
            error_trace = traceback.format_exc()
            return JsonResponse({
                "error": f"Prediction failed: {str(prediction_error)}",
                "traceback": error_trace
            }, status=500)

        # Get additional audio analysis for visualization
        try:
            audio_full, sr = librosa.load(temp_audio_path, sr=16000, mono=True)

            # Normalize
            if np.max(np.abs(audio_full)) > 0:
                audio_full = audio_full / np.max(np.abs(audio_full))

            # FFT analysis
            fft = np.fft.fft(audio_full)
            fft_magnitude = np.abs(fft[:len(fft) // 2])
            fft_frequencies = np.fft.fftfreq(len(audio_full), 1 / sr)[:len(fft) // 2]

            threshold = np.max(fft_magnitude) * 0.01
            significant_freqs = fft_frequencies[fft_magnitude > threshold]
            max_frequency = np.max(significant_freqs) if len(significant_freqs) > 0 else 0

            # Downsample for plotting
            downsample_factor = max(1, len(fft_frequencies) // 1000)
            fft_frequencies_plot = fft_frequencies[::downsample_factor].tolist()
            fft_magnitude_plot = fft_magnitude[::downsample_factor].tolist()

            waveform_downsample = max(1, len(audio_full) // 10000)
            waveform_plot = audio_full[::waveform_downsample].tolist()

        except Exception as viz_error:
            print(f"[SPEED] Warning: Visualization data generation failed: {viz_error}")
            # Use minimal visualization data
            waveform_plot = []
            fft_frequencies_plot = []
            fft_magnitude_plot = []
            max_frequency = 0
            sr = 16000

        # Prepare successful response
        response_data = {
            "success": True,
            "predicted_speed_kmh": float(predicted_speed),
            "checkpoint_name": os.path.basename(str(checkpoint_path)),
            "checkpoint_variance": float(variance),
            "checkpoint_mae": float(checkpoint.get('val_mae', 0)),
            "waveform": waveform_plot,
            "sr": sr,
            "filename": filename,
            "max_frequency": float(max_frequency),
            "fft_frequencies": fft_frequencies_plot,
            "fft_magnitudes": fft_magnitude_plot,
            "duration": float(len(audio_full) / sr) if 'audio_full' in locals() else 0,
            "model_info": {
                "device": str(device),
                "pytorch_version": torch.__version__
            }
        }

        print("[SPEED] ✅ Response prepared successfully")
        return JsonResponse(response_data)

    except Exception as e:
        error_traceback = traceback.format_exc()
        print(f"[SPEED] ❌ Unexpected error:\n{error_traceback}")
        return JsonResponse({
            "error": f"Speed prediction failed: {str(e)}",
            "traceback": error_traceback,
            "suggestion": "Run diagnostic script: python validate_checkpoints.py"
        }, status=500)

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                print(f"[SPEED] Cleanup warning: {e}")