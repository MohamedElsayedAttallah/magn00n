# signal_viewer_app/ecg_classification_utils.py
"""
ECG Classification Utilities for PTB-XL Dataset
Handles loading and inference of Keras/TensorFlow model for 12-lead ECG abnormality detection.
FIXED: Model expects raw ECG signal (1000, 12), not extracted features.
"""
import joblib #Loads the scaler for normalizing ECG signals
import numpy as np #Array operations
import tensorflow as tf
from tensorflow import keras #Deep learning framework for model loading and inference

# ============================================================================
# CLASS DEFINITIONS
# ============================================================================
# PTB-XL class names (5 superclasses)
# These represent the 5 major categories of ECG abnormalities the model can detect
CLASS_NAMES = [
    'NORM',  # Normal ECG - No abnormalities detected
    'MI',    # Myocardial Infarction - Heart attack evidence
    'STTC',  # ST/T Change - Abnormal repolarization
    'CD',    # Conduction Disturbance - Electrical conduction problems
    'HYP'    # Hypertrophy - Thickened heart muscle
]

CLASS_DESCRIPTIONS = {
    'NORM': 'Normal ECG - No abnormalities detected',
    'MI': 'Myocardial Infarction - Evidence of heart attack',
    'STTC': 'ST/T Change - Abnormalities in ST segment or T wave',
    'CD': 'Conduction Disturbance - Abnormal electrical conduction in the heart',
    'HYP': 'Hypertrophy - Thickening of heart muscle walls'
}

# ============================================================================
# MODEL LOADING
# ============================================================================
def load_ecg_model(model_path, device='cpu', num_classes=5, input_channels=12):
    """
       Load a Keras/TensorFlow model for ECG classification.

       The model is a multi-input architecture:
       - Input 0: 7 statistical features (mean, std, min, max, median, RMS, peak-to-peak)
       - Input 1: Raw 12-lead ECG signal (1000 samples × 12 channels)

       Args:
           model_path: Path to the .keras model file
           device: Not used for TensorFlow (kept for API compatibility) (TensorFlow auto-manages devices)
           num_classes: Number of output classes (default: 5 for PTB-XL superclasses)
           input_channels: Number of ECG leads (default: 12)

       Returns:
           Loaded Keras model ready for inference
    """
    try:
        print(f"[ECG Model] Loading Keras model from: {model_path}")

        # Load the Keras model without recompiling (faster for inference)
        model = keras.models.load_model(model_path, compile=False)

        print(f"[ECG Model] ✅ Model loaded successfully")
        print(f"[ECG Model] Model type: {type(model)}")
        print(f"[ECG Model] Number of inputs: {len(model.inputs)}")

        # Print detailed input information for debugging
        for i, inp in enumerate(model.inputs):
            print(f"[ECG Model]   Input {i}: {inp.name}, shape: {inp.shape}, dtype: {inp.dtype}")

        # Print output information
        for i, out in enumerate(model.outputs):
            print(f"[ECG Model]   Output {i}: {out.name}, shape: {out.shape}")

        return model

    except Exception as e:
        print(f"[ECG Model] ❌ Error loading model: {str(e)}")
        raise


# ============================================================================
# FEATURE EXTRACTION ( Extracts 7 statistical features from 12-lead ECG )
# ============================================================================
def extract_ecg_features(signal):
    """
        Extract 7 statistical features from 12-lead ECG signal.
        These features capture overall signal characteristics for Input 0 of the model.

        Features extracted (across all 12 leads):
        1. Mean heart rate (average signal amplitude)
        2. Standard deviation (signal variability)
        3. Min amplitude (lowest voltage)
        4. Max amplitude (highest voltage)
        5. Median amplitude (middle value)
        6. RMS (Root Mean Square - energy measure)
        7. Peak-to-peak amplitude (voltage range)

        Args:
            signal: numpy array of shape (num_channels, num_samples) - typically (12, 1000)

        Returns:
            numpy array of shape (7,) containing extracted features
    """
    print(f"[Feature Extraction] Input signal shape: {signal.shape}")

    # Flatten all channels to get overall signal statistics
    signal_flat = signal.flatten() # Converts (12, 1000) array to 1D array (12,000 values)
    # for aggregate feature computation

    # Extract 7 features (Computes 7 statistical features used as Input 0)
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


# ============================================================================
# FEATURE NORMALIZATION
# ============================================================================
def normalize_features(features):
    """
       Normalize features to reasonable range using Z-score normalization.
       This helps the model handle features of different scales.
       normalize_features: takes a list of 7 numbers (features) and rescales them so they are easier
       for a machine learning model to use.This process is called Z-score normalization.


       Process:
       1. Clip extreme outliers to [-10, 10]
       2. Apply Z-score normalization: (x - mean) / std
       3. Result centered around 0 with standard deviation of 1
       ( This calculation transforms the features so that the new list of numbers
        has an average of 0 and a standard deviation of 1. )

       Args:
           features: numpy array of shape (7,)

       Returns:
           Normalized features (same shape)

           The function cleans up the data by limiting extreme values and then rescales it,
            putting all features on the same scale (centered around 0) so
           the AI model can process them fairly without one feature overpowering another.
    """
    # Clip extreme values
    features = np.clip(features, -10, 10)

    # Z-score normalization
    mean = np.mean(features)
    std = np.std(features)

    if std > 0: #(The Normal Case) This means the numbers have some variation
        # (e.g., [1, 3, 5]). It's safe to divide by std, so the code runs the Z-score
        # calculation:
        features_normalized = (features - mean) / std
    else: #(The Special Case) This block runs if std is not greater than 0.
        # Since std can't be negative, this means std is exactly 0.
        #Why would std be 0? This only happens if all 7 of your features are the exact same number (e.g., [5, 5, 5, 5, 5, 5, 5]). There is zero variation.
        #What's the problem? If the code tried to run the calculation, it would be
        # (features - mean) / 0. Dividing by zero is mathematically impossible and would cause a ZeroDivisionError, crashing the program.
        features_normalized = features

    print(f"[Normalization] Normalized features: {features_normalized}")

    return features_normalized


# ============================================================================
# ECG PREDICTION
# ============================================================================
def predict_ecg_abnormality(model, signal, device='cpu'):
    """
       Predict ECG abnormality using trained model.

       MODEL IS MULTI-INPUT:
       - Input 0: 7 statistical features (shape: 1, 7)
       - Input 1: Raw signal (shape: 1, 1000, 12)

       Process:
       1. Extract and normalize 7 statistical features
       2. Prepare raw signal (pad/truncate to 1000 samples, transpose, scale)
       3. Pass both inputs to model
       4. Post-process predictions to get class probabilities

       Args:
           model: Loaded Keras model
           signal: numpy array of shape (num_channels, num_samples) - e.g., (12, 1000)
           device: Not used for TensorFlow (kept for API compatibility)

       Returns:
           Dictionary containing:
           - predicted_class: Class name (e.g., 'NORM', 'MI')
           - confidence: Probability of predicted class (0-1)
           - is_normal: Boolean indicating if prediction is 'NORM'
           - probabilities: Dict of all class probabilities
           - description: Human-readable description
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
        #This is a critical step. np.expand_dims adds a new dimension.
        #Why? The model expects a batch of inputs, not just a single input. This "1" represents a batch size of 1.
        # It's like putting a single sheet of paper into a box made for a stack.
        #.astype(np.float32) converts the numbers to the float32 format, which is standard for AI models.
        print(f"[ECG Prediction] Input 0 (Features) shape: {input_features.shape}")


        # --- Input 1: Prepare Raw Signal (1000, 12) ---
        print("[ECG Prediction] Preparing Input 1 (Raw Signal)...")

        # 1. Validate channels
        if signal.shape[0] != 12:
            raise ValueError(f"Expected 12 channels, got {signal.shape[0]}")

        num_samples = signal.shape[1]
        #Gets the length of the signal (e.g., 1000, or maybe 950, or 1100).
        target_samples = 1000
        #Defines the exact length the model needs: 1000 samples.

        # 2. Pad or truncate to 1000 samples
        if num_samples < target_samples:
            print(f"[ECG Prediction]   Padding signal from {num_samples} to {target_samples} samples.")
            # (In other words, don't change the 12 channels)
            #              |
            #              |         This second part corresponds to the 900 samples. It means "add 0 padding before the signal (at the start) and add 100 samples of padding after the signal (at the end)."
            #              |              | It means you are making the signal longer by adding 100 samples of silence (zeros) to the very end of it, without changing the start.
            pad_width = ((0, 0), (0, target_samples - num_samples))
            signal_processed = np.pad(signal, pad_width, 'constant', constant_values=0.0)
            #This command executes the plan. It fills those 100 new spots at the end of the
            # signal with the number 0.0 (which represents silence).
        elif num_samples > target_samples:
            print(f"[ECG Prediction]   Truncating signal from {num_samples} to {target_samples} samples.")
            signal_processed = signal[:, :target_samples]
        else:
            signal_processed = signal  # Shape is (12, 1000)

        # 3. Transpose from (Channels, Samples) to (Samples, Channels)
        signal_transposed = np.transpose(signal_processed, (1, 0))  # (1000, 12)
                #Why? Because this is the format the Keras model was trained on.

        # 4. Add batch dimension: (1000, 12) -> (1, 1000, 12)
        input_signal = np.expand_dims(signal_transposed, axis=0).astype(np.float32)
        #You add a batch dimension because AI models are designed to process data in groups
        # (or "batches") for efficiency, not just one item at a time.
        print(f"[ECG Prediction] Input 1 (Signal) shape: {input_signal.shape}")

        # --- End Input Prep ---

        # Check how many inputs the model expects
        num_model_inputs = len(model.inputs)
        print(f"[ECG Prediction] Model expects {num_model_inputs} input(s)")

        for i, inp in enumerate(model.inputs):
            print(f"[ECG Prediction]   Input {i}: shape={inp.shape}, dtype={inp.dtype}, name={inp.name}")

        # This is the most important step for getting an accurate prediction.
        # Load scaler (StandardScaler trained during model training)
        scaler = joblib.load("signal_viewer_app/assets/ecg_model/scaler.pkl")

        original_shape = input_signal.shape
        #aves the shape (1, 1000, 12) to use later.
        signal_flat = input_signal.reshape(-1, input_signal.shape[-1])
        #The scaler (from scikit-learn) expects a 2D array, not a 3D batch.
        # This line "flattens" the data from (1, 1000, 12) to (1000, 12).

        signal_scaled = scaler.transform(signal_flat)
        #This is the magic. It applies the exact same scaling from the training
        # data to your new signal. It normalizes each of the 12 channels.
        signal_scaled = signal_scaled.reshape(original_shape)
        input_signal = signal_scaled
        # Puts the scaled data back into the 3D shape the model needs: (1, 1000, 12).
        # Run inference
        # We now force the 2-input prediction

        if num_model_inputs == 2: #Checks that the loaded Keras model actually expects 2 inputs.
            print("[ECG Prediction] Running 2-input prediction (Features + Signal)...")
            predictions = model.predict([input_features, input_signal], verbose=0)
        else:
            # Fallback for any other configuration (e.g., 1 input, or 3+ inputs)
            print(f"[ECG Prediction] WARNING: Model expected 2 inputs, found {num_model_inputs}. Trying to proceed.")
            if num_model_inputs == 1:
                # This will likely fail based on the error, but it's a fallback.
                print("[ECG Prediction]   Trying to predict with features only...")
                predictions = model.predict(input_features, verbose=0)
            else: #This is a fallback in case the model structure is wrong (e.g., it only has 1 input).
                # It tries to run the prediction anyway, but it will likely fail or give bad results.
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
        if len(probabilities) < len(CLASS_NAMES):#A safety check. If the model output
            # 4 probabilities but your CLASS_NAMES list has 5, this code prevents a crash.
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
        #Finds the index (position) of the highest number. In [0.8, 0.1, 0.05, 0.0, 0.05],
        # the highest is 0.8 at index 0.
        predicted_class = CLASS_NAMES[predicted_class_idx]
        #Uses the index to get the class name from your list. CLASS_NAMES[0] is 'NORM'.
        confidence = float(probabilities[predicted_class_idx])
        #Gets the actual probability score for that class (e.g., 0.8).

        # Check if normal
        is_normal = (predicted_class == 'NORM')

        # Build probabilities dictionary
        probs_dict = {
            CLASS_NAMES[i]: float(probabilities[i])
            for i in range(len(CLASS_NAMES))
        }
        #A loop that builds a user-friendly dictionary,like : { 'NORM': 0.8, 'MI': 0.1, 'STTC': 0.05, ... }

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

        return result #Sends the result dictionary back to the detect_ecg_abnormality function in views.py.

    except Exception as e:
        print(f"[ECG Prediction] ❌ Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def get_class_description(class_name):
    """Get human-readable description for a class name."""
    return CLASS_DESCRIPTIONS.get(class_name, f"Unknown class: {class_name}")
    # If it fails to find it, it doesn't crash. Instead, it returns the default
    # value you provided as the second argument: f"Unknown class: {class_name}".

# Utility function to inspect model architecture
def inspect_model(model_path):
    """
    Inspect model architecture (useful for debugging).
    This function is a debugging tool for the developer.
     It is not used during the actual prediction
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