"""
CNN inference engine.

Rebuilds the EXACT custom architecture from the training notebook (Cell 10),
loads `best_model.weights.h5`, and runs preprocessing identical to the
training/eval pipeline (decode -> convert_image_dtype[0,1] -> resize 224).

If TensorFlow or the weights file is unavailable, the classifier drops into a
clearly-flagged DEMO MODE so the rest of the app (UI + LLM layer) stays testable.
"""
from __future__ import annotations

import hashlib
import io
import logging
from typing import Optional

import numpy as np

from app.config import Settings

logger = logging.getLogger("cnn")

# TensorFlow is heavy and optional at import time; guard it so the server can
# still boot (in demo mode) if it is missing.
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    _TF_AVAILABLE = True
    _TF_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover
    _TF_AVAILABLE = False
    _TF_IMPORT_ERROR = str(exc)


# --------------------------------------------------------------------------- #
#  Architecture — copied verbatim from the training notebook (Cell 10).
#  MUST stay byte-for-byte equivalent or load_weights() will not map correctly.
# --------------------------------------------------------------------------- #
def _conv_block(x, filters, kernel=3, pool=True):
    x = layers.Conv2D(filters, kernel, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if pool:
        x = layers.MaxPooling2D(2)(x)
    return x


def build_custom_cnn(input_shape=(224, 224, 3), num_classes=30, dropout_rate=0.45):
    inputs = keras.Input(shape=input_shape)
    x = _conv_block(inputs, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 256)
    x = _conv_block(x, 512, pool=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(num_classes)(x)
    outputs = layers.Activation("softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name="custom_cnn_deep")


class DiseaseClassifier:
    """Loads the CNN once and serves predictions + Grad-CAM inputs."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.img_size = settings.img_size
        self.class_names = settings.load_class_names()
        self.num_classes = len(self.class_names)

        self.model = None
        self.last_conv_layer_name: Optional[str] = None
        self.demo_mode = True
        self.status_detail = ""

        if not _TF_AVAILABLE:
            self.status_detail = f"TensorFlow not available ({_TF_IMPORT_ERROR}). Running in demo mode."
            logger.warning(self.status_detail)
            return

        try:
            self.model = build_custom_cnn(
                input_shape=(self.img_size, self.img_size, 3),
                num_classes=self.num_classes,
            )
            self.last_conv_layer_name = self._find_last_conv_layer()

            wpath = self.settings.weights_file
            if wpath.exists():
                self.model.load_weights(str(wpath))
                self.demo_mode = False
                self.status_detail = f"Loaded weights from {wpath.name}."
                logger.info(self.status_detail)
            else:
                self.status_detail = (
                    f"Weights file not found at {wpath}. Running in demo mode "
                    f"(place best_model.weights.h5 in models/ to enable real inference)."
                )
                logger.warning(self.status_detail)
        except Exception as exc:  # pragma: no cover
            self.model = None
            self.status_detail = f"Failed to build/load model: {exc}. Running in demo mode."
            logger.exception(self.status_detail)

    # ---------------- introspection ----------------
    def _find_last_conv_layer(self) -> Optional[str]:
        for layer in reversed(self.model.layers):
            # Conv2D layers have a 4-D output shape
            if isinstance(layer, layers.Conv2D):
                return layer.name
        return None

    def health(self) -> dict:
        return {
            "tensorflow_available": _TF_AVAILABLE,
            "demo_mode": self.demo_mode,
            "num_classes": self.num_classes,
            "last_conv_layer": self.last_conv_layer_name,
            "weights_path": str(self.settings.weights_file),
            "detail": self.status_detail,
        }

    # ---------------- preprocessing ----------------
    def prepare(self, image_bytes: bytes) -> np.ndarray:
        """Return a (1, H, W, 3) float32 array in [0, 1], matching training."""
        if _TF_AVAILABLE:
            img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, [self.img_size, self.img_size])
            arr = img.numpy().astype("float32")
        else:
            from PIL import Image

            pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil = pil.resize((self.img_size, self.img_size))
            arr = np.asarray(pil, dtype="float32") / 255.0
        return np.expand_dims(arr, axis=0)

    # ---------------- inference ----------------
    def predict(self, image_bytes: bytes, top_k: int = 5) -> dict:
        arr = self.prepare(image_bytes)
        if self.model is not None and not self.demo_mode:
            probs = self.model.predict(arr, verbose=0)[0].astype("float32")
        elif self.model is not None:
            # Model built but no weights: still produces (meaningless) outputs.
            probs = self._demo_probs(image_bytes)
        else:
            probs = self._demo_probs(image_bytes)

        order = np.argsort(probs)[::-1]
        predictions = [
            {
                "rank": int(r + 1),
                "label": self.class_names[int(i)],
                "probability": float(probs[int(i)]),
                "index": int(i),
            }
            for r, i in enumerate(order[:top_k])
        ]
        return {
            "demo_mode": self.demo_mode,
            "top_prediction": predictions[0],
            "predictions": predictions,
            "probabilities": probs.tolist(),
            "input_array": arr,  # reused by Grad-CAM (not serialized to client)
        }

    def _demo_probs(self, image_bytes: bytes) -> np.ndarray:
        """Deterministic pseudo-distribution seeded by image content."""
        seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=self.num_classes)
        logits[rng.integers(0, self.num_classes)] += 4.0  # a clear "winner"
        e = np.exp(logits - logits.max())
        return (e / e.sum()).astype("float32")
