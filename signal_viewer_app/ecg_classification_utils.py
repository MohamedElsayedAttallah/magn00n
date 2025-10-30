# signal_viewer_app/ecg_classification_utils.py
"""
ECG Classification Utilities for PTB-XL Dataset
Handles loading and inference of Keras/TensorFlow model for 12-lead ECG abnormality detection.
FIXED: Model expects raw ECG signal (1000, 12), not extracted features.
"""
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

# PTB-XL class names (5 superclasses)
CLASS_NAMES = [
    'NORM',  # Normal ECG
    'MI',  # Myocardial Infarction
    'STTC',  # ST/T Change
    'CD',  # Conduction Disturbance
    'HYP'  # Hypertrophy
]

CLASS_DESCRIPTIONS = {
    'NORM': 'Normal ECG - No abnormalities detected',
    'MI': 'Myocardial Infarction - Evidence of heart attack',
    'STTC': 'ST/T Change - Abnormalities in ST segment or T wave',
    'CD': 'Conduction Disturbance - Abnormal electrical conduction in the heart',
    'HYP': 'Hypertrophy - Thickening of heart muscle walls'
}


def load_ecg_model(model_path, device='cpu', num_classes=5, input_channels=12):
    """
    Load a Keras/TensorFlow model for ECG classification.

    Args:
        model_path: Path to the .keras model file
        device: Not used for TensorFlow (kept for API compatibility)
        num_classes: Number of output classes (default: 5 for PTB-XL superclasses)
        input_channels: Number of ECG leads (default: 12)

    Returns:
        Loaded Keras model
    """
    try:
        print(f"[ECG Model] Loading Keras model from: {model_path}")

        # Load the Keras model
        model = keras.models.load_model(model_path, compile=False)

        print(f"[ECG Model] ✅ Model loaded successfully")
        print(f"[ECG Model] Model type: {type(model)}")
        print(f"[ECG Model] Number of inputs: {len(model.inputs)}")

        # Print input information
        for i, inp in enumerate(model.inputs):
            print(f"[ECG Model]   Input {i}: {inp.name}, shape: {inp.shape}, dtype: {inp.dtype}")

        # Print output information
        for i, out in enumerate(model.outputs):
            print(f"[ECG Model]   Output {i}: {out.name}, shape: {out.shape}")

        return model

    except Exception as e:
        print(f"[ECG Model] ❌ Error loading model: {str(e)}")
        raise


def extract_ecg_features(signal):
    """
    Extract 7 statistical features from 12-lead ECG signal.

    Features extracted (across all 12 leads):
    1. Mean heart rate (average signal amplitude)
    2. Standard deviation (signal variability)
    3. Min amplitude
    4. Max amplitude
    5. Median amplitude
    6. RMS (Root Mean Square)
    7. Peak-to-peak amplitude

    Args:
        signal: numpy array of shape (num_channels, num_samples) - typically (12, 1000)

    Returns:
        numpy array of shape (7,) containing extracted features
    """
    print(f"[Feature Extraction] Input signal shape: {signal.shape}")

    # Flatten all channels to get overall signal statistics
    signal_flat = signal.flatten()

    # Extract 7 features
    features = np.array([
        np.mean(signal_flat),  # 1. Mean
        np.std(signal_flat),  # 2. Standard deviation
        np.min(signal_flat),  # 3. Minimum
        np.max(signal_flat),  # 4. Maximum
        np.median(signal_flat),  # 5. Median
        np.sqrt(np.mean(signal_flat ** 2)),  # 6. RMS (Root Mean Square)
        np.max(signal_flat) - np.min(signal_flat)  # 7. Peak-to-peak
    ], dtype=np.float32)

    print(f"[Feature Extraction] Extracted features: {features}")
    print(f"[Feature Extraction]   Mean: {features[0]:.4f}")
    print(f"[Feature Extraction]   Std: {features[1]:.4f}")
    print(f"[Feature Extraction]   Min: {features[2]:.4f}")
    print(f"[Feature Extraction]   Max: {features[3]:.4f}")
    print(f"[Feature Extraction]   Median: {features[4]:.4f}")
    print(f"[Feature Extraction]   RMS: {features[5]:.4f}")
    print(f"[Feature Extraction]   Peak-to-peak: {features[6]:.4f}")

    return features


def normalize_features(features):
    """
    Normalize features to reasonable range.

    Args:
        features: numpy array of shape (7,)

    Returns:
        Normalized features
    """
    # Clip extreme values
    features = np.clip(features, -10, 10)

    # Z-score normalization
    mean = np.mean(features)
    std = np.std(features)

    if std > 0:
        features_normalized = (features - mean) / std
    else:
        features_normalized = features

    print(f"[Normalization] Normalized features: {features_normalized}")

    return features_normalized


def predict_ecg_abnormality(model, signal, device='cpu'):
    """
    Predict ECG abnormality using trained model.
    MODEL IS MULTI-INPUT:
    - Input 0: 7 statistical features (shape 1, 7)
    - Input 1: Raw signal (shape 1, 1000, 12)

    Args:
        model: Loaded Keras model
        signal: numpy array of shape (num_channels, num_samples) - e.g., (12, 1000)
        device: Not used for TensorFlow (kept for API compatibility)

    Returns:
        Dictionary containing prediction results
    """
    try:
        print(f"[ECG Prediction] Starting prediction for multi-input model...")
        print(f"[ECG Prediction] Input signal shape: {signal.shape}")  # (12, N)

        # --- Input 0: Prepare 7 Features ---
        print("[ECG Prediction] Preparing Input 0 (Features)...")
        features = extract_ecg_features(signal)
        features_normalized = normalize_features(features)
        # Add batch dimension: (7,) -> (1, 7)
        input_features = np.expand_dims(features_normalized, axis=0).astype(np.float32)
        print(f"[ECG Prediction] Input 0 (Features) shape: {input_features.shape}")

        # --- Input 1: Prepare Raw Signal (1000, 12) ---
        print("[ECG Prediction] Preparing Input 1 (Raw Signal)...")

        # 1. Validate channels
        if signal.shape[0] != 12:
            raise ValueError(f"Expected 12 channels, got {signal.shape[0]}")

        num_samples = signal.shape[1]
        target_samples = 1000

        # 2. Pad or truncate to 1000 samples
        if num_samples < target_samples:
            print(f"[ECG Prediction]   Padding signal from {num_samples} to {target_samples} samples.")
            pad_width = ((0, 0), (0, target_samples - num_samples))
            signal_processed = np.pad(signal, pad_width, 'constant', constant_values=0.0)
        elif num_samples > target_samples:
            print(f"[ECG Prediction]   Truncating signal from {num_samples} to {target_samples} samples.")
            signal_processed = signal[:, :target_samples]
        else:
            signal_processed = signal  # Shape is (12, 1000)

        # 3. Transpose from (Channels, Samples) to (Samples, Channels)
        signal_transposed = np.transpose(signal_processed, (1, 0))  # (1000, 12)

        # 4. Add batch dimension: (1000, 12) -> (1, 1000, 12)
        input_signal = np.expand_dims(signal_transposed, axis=0).astype(np.float32)
        print(f"[ECG Prediction] Input 1 (Signal) shape: {input_signal.shape}")

        # --- End Input Prep ---

        # Check how many inputs the model expects
        num_model_inputs = len(model.inputs)
        print(f"[ECG Prediction] Model expects {num_model_inputs} input(s)")

        for i, inp in enumerate(model.inputs):
            print(f"[ECG Prediction]   Input {i}: shape={inp.shape}, dtype={inp.dtype}, name={inp.name}")

        scaler = joblib.load("signal_viewer_app/assets/ecg_model/scaler.pkl")
        original_shape = input_signal.shape
        signal_flat = input_signal.reshape(-1, input_signal.shape[-1])
        signal_scaled = scaler.transform(signal_flat)
        signal_scaled = signal_scaled.reshape(original_shape)
        input_signal = signal_scaled
        # Run inference
        # We now force the 2-input prediction
        if num_model_inputs == 2:
            print("[ECG Prediction] Running 2-input prediction (Features + Signal)...")
            predictions = model.predict([input_features, input_signal], verbose=0)
        else:
            # Fallback for any other configuration (e.g., 1 input, or 3+ inputs)
            print(f"[ECG Prediction] WARNING: Model expected 2 inputs, found {num_model_inputs}. Trying to proceed.")
            if num_model_inputs == 1:
                # This will likely fail based on the error, but it's a fallback.
                print("[ECG Prediction]   Trying to predict with features only...")
                predictions = model.predict(input_features, verbose=0)
            else:
                # Try to send the two inputs we prepared, plus dummy data for the rest
                print("[ECG Prediction]   Trying to predict with (Features, Signal, Dummies)...")
                inputs = [input_features, input_signal]
                for i in range(2, num_model_inputs):
                    dummy = np.zeros((1, 2), dtype=np.float32)  # Default dummy
                    inputs.append(dummy)
                predictions = model.predict(inputs, verbose=0)

        print(f"[ECG Prediction] Raw predictions shape: {predictions.shape}")
        print(f"[ECG Prediction] Raw predictions: {predictions[0]}")

        # Get probabilities
        probabilities = predictions[0]

        # Handle different output formats
        if len(probabilities) < len(CLASS_NAMES):
            print(
                f"[ECG Prediction] Warning: Model output size ({len(probabilities)}) < CLASS_NAMES ({len(CLASS_NAMES)})")
            # Pad with zeros
            padded_probs = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            padded_probs[:len(probabilities)] = probabilities
            probabilities = padded_probs
        elif len(probabilities) > len(CLASS_NAMES):
            print(
                f"[ECG Prediction] Warning: Model output size ({len(probabilities)}) > CLASS_NAMES ({len(CLASS_NAMES)})")
            # Use only first N probabilities
            probabilities = probabilities[:len(CLASS_NAMES)]

        # Get predicted class
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(probabilities[predicted_class_idx])

        # Check if normal
        is_normal = (predicted_class == 'NORM')

        # Build probabilities dictionary
        probs_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }

        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'is_normal': is_normal,
            'probabilities': probs_dict,
            'description': get_class_description(predicted_class)
        }

        print(f"[ECG Prediction] ✅ Prediction complete")
        print(f"[ECG Prediction] Predicted class: {predicted_class}")
        print(f"[ECG Prediction] Confidence: {confidence:.2%}")
        print(f"[ECG Prediction] Is normal: {is_normal}")

        return result

    except Exception as e:
        print(f"[ECG Prediction] ❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_class_description(class_name):
    """Get human-readable description for a class name."""
    return CLASS_DESCRIPTIONS.get(class_name, f"Unknown class: {class_name}")


# Utility function to inspect model architecture
def inspect_model(model_path):
    """
    Inspect model architecture (useful for debugging).

    Args:
        model_path: Path to the .keras model file
    """
    try:
        model = keras.models.load_model(model_path, compile=False)

        print("\n" + "=" * 60)
        print("MODEL ARCHITECTURE INSPECTION")
        print("=" * 60)

        print("\nINPUTS:")
        for i, inp in enumerate(model.inputs):
            print(f"  Input {i}:")
            print(f"    Name: {inp.name}")
            print(f"    Shape: {inp.shape}")
            print(f"    dtype: {inp.dtype}")

        print("\nOUTPUTS:")
        for i, out in enumerate(model.outputs):
            print(f"  Output {i}:")
            print(f"    Name: {out.name}")
            print(f"    Shape: {out.shape}")

        print("\nMODEL SUMMARY:")
        model.summary()

        print("=" * 60 + "\n")

        # Test with dummy data
        print("\nTESTING WITH DUMMY DATA:")
        dummy_features = np.random.randn(1, 7).astype(np.float32)
        print(f"Dummy input shape: {dummy_features.shape}")

        try:
            result = model.predict(dummy_features, verbose=0)
            print(f"✅ Prediction successful!")
            print(f"Output shape: {result.shape}")
            print(f"Output: {result[0]}")
        except Exception as e:
            print(f"❌ Prediction failed: {e}")

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"Error inspecting model: {e}")
        import traceback
        traceback.print_exc()