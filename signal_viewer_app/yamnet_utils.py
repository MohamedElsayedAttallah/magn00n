"""
YAMNet model utilities for fine-tuning and inference.
FIXED VERSION - Proper model loading and prediction handling
"""

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from typing import Tuple, List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store YAMNet model
_yamnet_model_instance = None

def initialize_yamnet():
    """Initialize YAMNet model and store it globally."""
    global _yamnet_model_instance

    if _yamnet_model_instance is not None:
        logger.info("YAMNet already initialized")
        return _yamnet_model_instance

    try:
        # Set TF Hub cache directory
        package_dir = os.path.dirname(__file__)
        cache_dir = os.path.abspath(os.path.join(package_dir, 'cache'))
        os.makedirs(cache_dir, exist_ok=True)
        os.environ['TFHUB_CACHE_DIR'] = cache_dir

        logger.info("Loading YAMNet from TensorFlow Hub...")
        yamnet_url = 'https://tfhub.dev/google/yamnet/1'
        _yamnet_model_instance = hub.load(yamnet_url)

        # Test the model
        test_input = tf.zeros(16000, dtype=tf.float32)
        _ = _yamnet_model_instance(test_input)

        logger.info("✅ YAMNet initialized successfully")
        return _yamnet_model_instance

    except Exception as e:
        logger.error(f"❌ Error initializing YAMNet: {str(e)}")
        raise


def extract_yamnet_embeddings(audio_batch):
    """
    Extract YAMNet embeddings for a batch of audio samples.

    Args:
        audio_batch: Batch of audio samples with shape (batch_size, num_samples)

    Returns:
        Embeddings with shape (batch_size, num_frames, 1024)
    """
    global _yamnet_model_instance

    if _yamnet_model_instance is None:
        initialize_yamnet()

    # Process each sample in the batch
    def process_single_audio(audio):
        """Process a single audio sample."""
        # YAMNet expects 1D audio
        embeddings, _, _ = _yamnet_model_instance(audio)
        return embeddings

    # Use map_fn to process batch
    embeddings = tf.map_fn(
        process_single_audio,
        audio_batch,
        dtype=tf.float32,
        parallel_iterations=1
    )

    return embeddings


class YAMNetEmbeddingLayer(tf.keras.layers.Layer):
    """Custom Keras layer for YAMNet embeddings."""

    def __init__(self, **kwargs):
        super(YAMNetEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Initialize YAMNet when layer is built
        initialize_yamnet()
        super(YAMNetEmbeddingLayer, self).build(input_shape)

    def call(self, inputs):
        """Extract embeddings from audio."""
        return extract_yamnet_embeddings(inputs)

    def get_config(self):
        config = super(YAMNetEmbeddingLayer, self).get_config()
        return config


def load_finetuned_model(model_path):
    """
    Load fine-tuned YAMNet model with proper error handling.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    logger.info(f"Loading model from {model_path}...")

    # Initialize YAMNet first
    initialize_yamnet()

    # Define custom objects
    custom_objects = {
        'YAMNetEmbeddingLayer': YAMNetEmbeddingLayer,
        'extract_yamnet_embeddings': extract_yamnet_embeddings,
    }

    try:
        # Try loading with compile=False first (recommended for inference)
        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False
        )
        logger.info("✅ Model loaded successfully")

        # Print model summary for debugging
        logger.info("Model architecture:")
        model.summary(print_fn=logger.info)

        return model

    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.info("Attempting to rebuild model architecture...")

        try:
            # If loading fails, try to rebuild the architecture
            model = rebuild_model_architecture()

            # Try to load weights only
            model.load_weights(model_path)
            logger.info("✅ Model weights loaded successfully")

            return model

        except Exception as e2:
            logger.error(f"❌ Failed to rebuild model: {str(e2)}")
            raise RuntimeError(
                f"Could not load model. Original error: {str(e)}, "
                f"Rebuild error: {str(e2)}"
            )


def rebuild_model_architecture(num_classes=3):
    """
    Rebuild the fine-tuned model architecture from scratch.

    Args:
        num_classes: Number of output classes

    Returns:
        Keras model with architecture matching the trained model
    """
    logger.info("Rebuilding model architecture...")

    # Initialize YAMNet
    initialize_yamnet()

    # Input layer
    audio_input = tf.keras.layers.Input(
        shape=(None,),
        dtype=tf.float32,
        name='audio_input'
    )

    # YAMNet embeddings
    yamnet_embeddings = YAMNetEmbeddingLayer(
        name='yamnet_embeddings'
    )(audio_input)

    # Global average pooling
    pooled = tf.keras.layers.GlobalAveragePooling1D(
        name='global_avg_pool'
    )(yamnet_embeddings)

    # Dropout
    dropout1 = tf.keras.layers.Dropout(0.5, name='dropout')(pooled)

    # Dense layers
    dense1 = tf.keras.layers.Dense(
        512,
        activation='relu',
        name='dense1'
    )(dropout1)

    dropout2 = tf.keras.layers.Dropout(0.3, name='dropout2')(dense1)

    dense2 = tf.keras.layers.Dense(
        256,
        activation='relu',
        name='dense2'
    )(dropout2)

    # Output layer
    output = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        name='predictions'
    )(dense2)

    # Create model
    model = tf.keras.Model(
        inputs=audio_input,
        outputs=output,
        name='yamnet_finetuned'
    )

    logger.info("✅ Model architecture rebuilt")

    return model


def predict_audio_class(model, audio_array, class_names=['Drone', 'Bird', 'Noise/Other']):
    """
    Make prediction on audio sample with proper preprocessing.

    Args:
        model: Loaded Keras model
        audio_array: Audio numpy array (1D)
        class_names: List of class names

    Returns:
        Dictionary with prediction results
    """
    logger.info(f"Making prediction on audio with {len(audio_array)} samples")

    # Ensure audio is float32
    audio = audio_array.astype(np.float32)

    # Normalize audio to [-1, 1] range
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    # Add batch dimension
    audio_batch = np.expand_dims(audio, axis=0)

    logger.info(f"Audio batch shape: {audio_batch.shape}")
    logger.info(f"Audio range: [{np.min(audio):.3f}, {np.max(audio):.3f}]")

    # Make prediction
    try:
        predictions = model.predict(audio_batch, verbose=0)
        logger.info(f"Raw predictions: {predictions}")

        # Get probabilities
        probabilities = predictions[0]

        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx]

        logger.info(f"✅ Predicted: {predicted_class} ({confidence:.2%} confidence)")

        return {
            'predicted_class': predicted_class,
            'predicted_index': int(predicted_idx),
            'confidence': float(confidence),
            'probabilities': probabilities.tolist(),
            'class_names': class_names
        }

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


class YAMNetPreprocessor:
    """Handles audio preprocessing for YAMNet."""

    @staticmethod
    def preprocess_audio(audio, target_sr=16000, normalize=True):
        """
        Preprocess audio for YAMNet input.

        Args:
            audio: Audio numpy array
            target_sr: Target sample rate (YAMNet expects 16kHz)
            normalize: Whether to normalize audio

        Returns:
            Preprocessed audio array
        """
        # Ensure float32
        audio = audio.astype(np.float32)

        # Normalize to [-1, 1] if requested
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val

        return audio

    @staticmethod
    def ensure_length(audio, target_length=16000):
        """
        Ensure audio has target length by padding or truncating.

        Args:
            audio: Audio array
            target_length: Target length in samples

        Returns:
            Audio array with target length
        """
        if len(audio) < target_length:
            # Pad with zeros
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        elif len(audio) > target_length:
            # Truncate
            audio = audio[:target_length]

        return audio