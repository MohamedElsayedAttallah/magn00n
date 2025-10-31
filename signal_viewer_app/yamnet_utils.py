"""
YAMNet model utilities for fine-tuning and inference.

YAMNet is a pretrained deep learning model from Google that classifies audio into 521 categories.
This module provides utilities for:
1. Loading and initializing YAMNet from TensorFlow Hub
2. Extracting embeddings (feature vectors) from audio
3. Fine-tuning YAMNet for custom audio classification tasks
4. Making predictions on audio samples

FIXED VERSION - Proper model loading and prediction handling
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import Tuple, List
import logging
import os

# ==================== LOGGING CONFIGURATION ====================
# Configure logging to display INFO level messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== GLOBAL MODEL INSTANCE ====================
# Store YAMNet model globally to avoid reloading it multiple times
# This improves performance as loading from TensorFlow Hub is expensive
_yamnet_model_instance = None


def initialize_yamnet():
    """
    Initialize YAMNet model from TensorFlow Hub and store it globally.
    
    YAMNet is loaded from Google's TensorFlow Hub repository. Once loaded,
    it's cached both on disk (in TFHUB_CACHE_DIR) and in memory (global variable)
    for efficient reuse.
    
    Returns:
        The loaded YAMNet model instance
        
    Raises:
        Exception if model fails to load or test prediction fails
    """
    global _yamnet_model_instance

    # -------------------- Check if Already Initialized --------------------
    # If model is already loaded in memory, return it immediately
    if _yamnet_model_instance is not None:
        logger.info("YAMNet already initialized")
        return _yamnet_model_instance

    try:
        # -------------------- Set Cache Directory --------------------
        # Configure where TensorFlow Hub downloads and caches models
        # Store in a 'cache' folder relative to this file
        package_dir = os.path.dirname(__file__)
        cache_dir = os.path.abspath(os.path.join(package_dir, 'cache'))
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TFHUB_CACHE_DIR'] = cache_dir

        # -------------------- Load YAMNet from TensorFlow Hub --------------------
        logger.info("Loading YAMNet from TensorFlow Hub...")
        yamnet_url = 'https://tfhub.dev/google/yamnet/1'
        _yamnet_model_instance = hub.load(yamnet_url)
        # This downloads the model (~5MB) on first run, then uses cached version

        # -------------------- Test the Model --------------------
        # Verify model works by running a test prediction with dummy audio
        # 16000 samples = 1 second of audio at 16kHz (YAMNet's expected rate)
        test_input = tf.zeros(16000, dtype=tf.float32)
        _ = _yamnet_model_instance(test_input)
        # If this fails, an exception will be raised

        logger.info("✅ YAMNet initialized successfully")
        return _yamnet_model_instance

    except Exception as e:
        logger.error(f"❌ Error initializing YAMNet: {str(e)}")
        raise


def extract_yamnet_embeddings(audio_batch):
    """
    Extract YAMNet embeddings (feature vectors) for a batch of audio samples.
    
    YAMNet converts audio into 1024-dimensional embeddings for each 0.96-second frame.
    These embeddings capture audio characteristics and can be used for:
    - Audio classification
    - Feature extraction for custom models
    - Transfer learning
    
    Args:
        audio_batch: Batch of audio samples with shape (batch_size, num_samples)
                    Each sample should be 1D audio at 16kHz
    
    Returns:
        Embeddings tensor with shape (batch_size, num_frames, 1024)
        where num_frames depends on audio length (~1 frame per 0.96 seconds)
    """
    global _yamnet_model_instance

    # -------------------- Ensure YAMNet is Initialized --------------------
    if _yamnet_model_instance is None:
        initialize_yamnet()

    # -------------------- Process Each Audio Sample --------------------
    def process_single_audio(audio):
        """
        Process a single audio sample through YAMNet.
        
        YAMNet returns three outputs:
        1. embeddings: 1024-dimensional feature vectors for each frame
        2. scores: Probabilities for 521 audio classes
        3. spectrogram: Mel spectrogram used internally
        
        We only need the embeddings for feature extraction.
        """
        # YAMNet expects 1D audio (mono, 16kHz)
        embeddings, _, _ = _yamnet_model_instance(audio)
        return embeddings

    # -------------------- Batch Processing --------------------
    # Use TensorFlow's map_fn to process each audio in the batch
    # parallel_iterations=1 ensures sequential processing (safer for large models)
    embeddings = tf.map_fn(
        process_single_audio,
        audio_batch,
        dtype=tf.float32,
        parallel_iterations=1
    )

    return embeddings


class YAMNetEmbeddingLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer that wraps YAMNet embedding extraction.
    
    This allows YAMNet to be used as a layer in a Keras model, enabling:
    - End-to-end training with YAMNet as a feature extractor
    - Building custom models on top of YAMNet embeddings
    - Saving/loading models that include YAMNet
    
    Usage:
        input = tf.keras.layers.Input(shape=(None,))
        embeddings = YAMNetEmbeddingLayer()(input)
        output = tf.keras.layers.Dense(num_classes)(embeddings)
    """

    def __init__(self, **kwargs):
        """Initialize the layer."""
        super(YAMNetEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Build the layer - called automatically when layer is first used.
        
        This is where we initialize YAMNet to ensure it's loaded before
        the layer is used for the first time.
        """
        # Initialize YAMNet when layer is built (lazy initialization)
        initialize_yamnet()
        super(YAMNetEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass: extract embeddings from audio inputs.
        
        Args:
            inputs: Audio tensor with shape (batch_size, num_samples)
            
        Returns:
            Embeddings tensor with shape (batch_size, num_frames, 1024)
        """
        return extract_yamnet_embeddings(inputs)

    def get_config(self):
        """
        Get layer configuration for serialization.
        
        This allows the layer to be saved and loaded with Keras models.
        """
        config = super(YAMNetEmbeddingLayer, self).get_config()
        return config


def load_finetuned_model(model_path):
    """
    Load a fine-tuned YAMNet model with proper error handling and fallback.
    
    Fine-tuning means taking the pretrained YAMNet and training additional layers
    on top for a specific task (e.g., drone vs. bird detection). This function
    handles loading such models, dealing with common issues like:
    - Missing custom objects (YAMNetEmbeddingLayer)
    - Serialization issues
    - Architecture mismatches
    
    Args:
        model_path: Path to the saved Keras model file (.h5 or SavedModel format)
    
    Returns:
        Loaded Keras model ready for inference
        
    Raises:
        RuntimeError if model cannot be loaded with either method
    """
    # -------------------- Log Load Attempt --------------------
    logger.info(f"Loading model from {model_path}...")

    # -------------------- Initialize YAMNet --------------------
    # Ensure YAMNet runtime is ready before loading the model
    # (Required because the model uses YAMNetEmbeddingLayer)
    initialize_yamnet()

    # -------------------- Define Custom Objects --------------------
    # Custom objects are non-standard layers/functions used in the model
    # Keras needs to know about these to deserialize the model correctly
    custom_objects = {
        'YAMNetEmbeddingLayer': YAMNetEmbeddingLayer,
        'extract_yamnet_embeddings': extract_yamnet_embeddings,
    }

    try:
        # -------------------- Attempt to Load Full Model --------------------
        # Load the complete model including architecture and weights
        # compile=False means we don't need to restore optimizer state
        # (faster and sufficient for inference-only use)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False  # Skip compilation for inference
        )
        
        # -------------------- Log Success --------------------
        logger.info("✅ Model loaded successfully")

        # Print model architecture for debugging/verification
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)

        return model

    except Exception as e:
        # -------------------- Fallback: Rebuild Architecture --------------------
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.info("Attempting to rebuild model architecture...")

        try:
            # If full model loading fails, try to:
            # 1. Rebuild the architecture from scratch
            # 2. Load only the weights into the rebuilt architecture
            model = rebuild_model_architecture()

            # Load just the weights (not the full model structure)
            model.load_weights(model_path)
            logger.info("✅ Model weights loaded successfully")

            return model

        except Exception as e2:
            # -------------------- Both Methods Failed --------------------
            logger.error(f"❌ Failed to rebuild model: {str(e2)}")
            raise RuntimeError(
                f"Could not load model. Original error: {str(e)}, "
                f"Rebuild error: {str(e2)}"
            )


def rebuild_model_architecture(num_classes=3):
    """
    Rebuild the fine-tuned model architecture from scratch.
    
    This is a fallback mechanism when the saved model file is corrupted or
    incompatible. It recreates the exact architecture used during training,
    allowing us to load just the weights.
    
    The architecture:
    1. YAMNet embeddings (1024-dim features per frame)
    2. Global average pooling (aggregate all frames)
    3. Dense layers with dropout (classification head)
    4. Softmax output (class probabilities)
    
    Args:
        num_classes: Number of output classes (default: 3 for Drone/Bird/Noise)
    
    Returns:
        Keras model with architecture matching the trained model (but no weights)
    """
    logger.info("Rebuilding model architecture...")

    # -------------------- Initialize YAMNet --------------------
    initialize_yamnet()

    # -------------------- Input Layer --------------------
    # Accepts variable-length audio (None means any length)
    # Shape (None,) represents 1D audio of any length
    audio_input = tf.keras.layers.Input(
        shape=(None,),
        dtype=tf.float32,
        name='audio_input'
    )

    # -------------------- YAMNet Embedding Layer --------------------
    # Extract 1024-dimensional embeddings for each audio frame
    # Output shape: (batch_size, num_frames, 1024)
    yamnet_embeddings = YAMNetEmbeddingLayer(
        name='yamnet_embeddings'
    )(audio_input)

    # -------------------- Global Average Pooling --------------------
    # Aggregate embeddings from all frames into a single vector
    # Converts (batch_size, num_frames, 1024) → (batch_size, 1024)
    # This creates a fixed-size representation regardless of audio length
    pooled = tf.keras.layers.GlobalAveragePooling1D(
        name='global_avg_pool'
    )(yamnet_embeddings)

    # -------------------- Classification Head --------------------
    # First dropout layer (50% dropout rate for regularization)
    dropout1 = tf.keras.layers.Dropout(0.5, name='dropout')(pooled)

    # First dense layer (512 units with ReLU activation)
    dense1 = tf.keras.layers.Dense(
        512,
        activation='relu',
        name='dense1'
    )(dropout1)

    # Second dropout layer (30% dropout rate)
    dropout2 = tf.keras.layers.Dropout(0.3, name='dropout2')(dense1)

    # Second dense layer (256 units with ReLU activation)
    dense2 = tf.keras.layers.Dense(
        256,
        activation='relu',
        name='dense2'
    )(dropout2)

    # -------------------- Output Layer --------------------
    # Final layer with softmax activation for class probabilities
    # Outputs probabilities for each class (sum to 1.0)
    output = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(dense2)

    # -------------------- Create Model --------------------
    model = tf.keras.Model(
        inputs=audio_input,
        outputs=output,
        name='yamnet_finetuned'
    )

    logger.info("✅ Model architecture rebuilt")

    return model


def predict_audio_class(model, audio_array, class_names=['Drone', 'Bird', 'Noise/Other']):
    """
    Make a classification prediction on an audio sample.
    
    This function:
    1. Preprocesses the audio (normalize, ensure correct format)
    2. Runs the model to get class probabilities
    3. Determines the predicted class and confidence
    4. Returns structured prediction results
    
    Args:
        model: Loaded Keras model (from load_finetuned_model)
        audio_array: 1D numpy array of audio samples (any length, mono, preferably 16kHz)
        class_names: List of class labels (default: ['Drone', 'Bird', 'Noise/Other'])
    
    Returns:
        Dictionary containing:
        - predicted_class: Name of predicted class
        - predicted_index: Index of predicted class
        - confidence: Probability of predicted class (0-1)
        - probabilities: List of probabilities for all classes
        - class_names: List of class names
    
    Raises:
        Exception if prediction fails (logged with traceback)
    """
    logger.info(f"Making prediction on audio with {len(audio_array)} samples")

    # -------------------- Preprocess Audio --------------------
    # Ensure audio is float32 (required by TensorFlow)
    audio = audio_array.astype(np.float32)

    # Normalize audio to [-1, 1] range to prevent numerical issues
    # This ensures consistent input scale regardless of original audio amplitude
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Add batch dimension: (samples,) → (1, samples)
    # Models expect batched input even for single samples
    audio_batch = np.expand_dims(audio, axis=0)

    # -------------------- Log Input Statistics --------------------
    logger.info(f"Audio batch shape: {audio_batch.shape}")
    logger.info(f"Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")

    # -------------------- Make Prediction --------------------
    try:
        # Run model inference
        # predictions shape: (1, num_classes) - probabilities for each class
        predictions = model.predict(audio_batch, verbose=0)
        logger.info(f"Raw predictions: {predictions}")

        # -------------------- Process Predictions --------------------
        # Get probabilities for the single sample (remove batch dimension)
        probabilities = predictions[0]

        # Find class with highest probability
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx]

        logger.info(f"✅ Predicted: {predicted_class} ({confidence:.2%} confidence)")

        # -------------------- Return Results --------------------
        return {
            'predicted_class': predicted_class,        # e.g., "Drone"
            'predicted_index': int(predicted_idx),     # e.g., 0
            'confidence': float(confidence),           # e.g., 0.87
            'probabilities': probabilities.tolist(),   # e.g., [0.87, 0.10, 0.03]
            'class_names': class_names                 # e.g., ['Drone', 'Bird', 'Noise/Other']
        }

    except Exception as e:
        # -------------------- Error Handling --------------------
        logger.error(f"❌ Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


class YAMNetPreprocessor:
    """
    Handles audio preprocessing for YAMNet input.
    
    YAMNet has specific requirements:
    - Audio must be 16kHz sample rate
    - Audio should be normalized to [-1, 1]
    - Audio should be mono (single channel)
    
    This class provides utilities to prepare audio for YAMNet.
    """

    @staticmethod
    def preprocess_audio(audio, target_sr=16000, normalize=True):
        """
        Preprocess audio for YAMNet input with proper formatting and normalization.
        
        Args:
            audio: Audio numpy array (1D for mono, 2D for stereo)
            target_sr: Target sample rate (YAMNet expects 16kHz)
            normalize: Whether to normalize audio to [-1, 1] range
        
        Returns:
            Preprocessed audio array (float32, normalized)
        
        Note:
            This function does NOT resample audio. Use librosa.resample()
            before calling this if your audio is not at 16kHz.
        """
        # -------------------- Convert to Float32 --------------------
        # Ensure audio is in float32 format (required by TensorFlow)
        audio = audio.astype(np.float32)

        # -------------------- Normalize Audio --------------------
        # Scale audio to [-1, 1] range if requested
        # This prevents clipping and ensures consistent input scale
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        return audio

    @staticmethod
    def ensure_length(audio, target_length=16000):
        """
        Ensure audio has exact target length by padding or truncating.
        
        This is useful when you need fixed-length audio inputs, for example:
        - Training with fixed batch sizes
        - Comparing audio segments of equal length
        - Ensuring minimum duration for analysis
        
        Args:
            audio: Audio array (1D)
            target_length: Target length in samples (default: 16000 = 1 second at 16kHz)
        
        Returns:
            Audio array with exactly target_length samples
        """
        # -------------------- Pad if Too Short --------------------
        if len(audio) < target_length:
            # Add zeros to the end to reach target length
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            # mode='constant' pads with zeros (silence)
        
        # -------------------- Truncate if Too Long --------------------
        elif len(audio) > target_length:
            # Keep only the first target_length samples
            audio = audio[:target_length]

        return audio