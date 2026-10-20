"""
Grad-CAM heatmaps for the custom CNN.

Produces:
  - a base64 PNG overlay (heatmap superimposed on the original image), and
  - a plain-English description of WHERE and HOW concentrated the CNN's
    attention was, which is fed to the LLM layer as visual evidence.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("gradcam")

try:
    import tensorflow as tf
    from tensorflow import keras

    _TF_AVAILABLE = True
except Exception:  # pragma: no cover
    _TF_AVAILABLE = False


def _compute_heatmap(model, img_array: np.ndarray, last_conv_layer_name: str,
                     pred_index: Optional[int] = None) -> Optional[np.ndarray]:
    """Standard Grad-CAM using pre-softmax logits for cleaner gradients."""
    if not _TF_AVAILABLE or model is None or last_conv_layer_name is None:
        return None
    try:
        conv_layer = model.get_layer(last_conv_layer_name)
        # model.layers[-2] is the Dense(num_classes) logits (layer before softmax).
        logits_output = model.layers[-2].output
        grad_model = keras.models.Model(model.inputs, [conv_layer.output, logits_output])

        arr = tf.convert_to_tensor(img_array)
        with tf.GradientTape() as tape:
            conv_out, logits = grad_model(arr, training=False)
            if pred_index is None:
                pred_index = int(tf.argmax(logits[0]))
            class_channel = logits[:, pred_index]

        grads = tape.gradient(class_channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0.0)
        maxv = tf.reduce_max(heatmap)
        heatmap = heatmap / (maxv + 1e-8)
        return heatmap.numpy().astype("float32")
    except Exception as exc:  # pragma: no cover
        logger.warning("Grad-CAM failed: %s", exc)
        return None


def _jet(gray: np.ndarray) -> np.ndarray:
    """Approximate the 'jet' colormap. Input HxW in [0,1] -> HxWx3 uint8."""
    v = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype("uint8")


def _describe_focus(heatmap: np.ndarray) -> str:
    """Turn the heatmap into a short textual description for the LLM prompt."""
    h, w = heatmap.shape
    thresh = 0.5
    hot = heatmap >= thresh
    coverage = float(hot.mean())

    if hot.sum() == 0:
        return "diffuse, low-intensity activation with no single dominant focus"

    ys, xs = np.where(hot)
    cy, cx = ys.mean() / h, xs.mean() / w
    vert = "upper" if cy < 0.4 else ("lower" if cy > 0.6 else "central")
    horiz = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
    region = "central" if (vert == "central" and horiz == "center") else f"{vert}-{horiz}".strip("-")

    if coverage < 0.08:
        spread = "a tightly localized focal region"
    elif coverage < 0.25:
        spread = "a moderately sized region"
    else:
        spread = "a broad, distributed region"

    return f"{spread} in the {region} portion of the image (~{coverage * 100:.0f}% high-salience area)"


def generate(model, img_array: np.ndarray, original_bytes: bytes, img_size: int,
             last_conv_layer_name: str, pred_index: Optional[int] = None,
             alpha: float = 0.42) -> dict:
    """
    Returns {"overlay_png_b64": str|None, "focus_description": str, "available": bool}.
    """
    heatmap = _compute_heatmap(model, img_array, last_conv_layer_name, pred_index)
    if heatmap is None:
        return {
            "overlay_png_b64": None,
            "focus_description": "Grad-CAM unavailable (running without a loaded model).",
            "available": False,
        }

    # Original image resized to model input size.
    base = Image.open(io.BytesIO(original_bytes)).convert("RGB").resize((img_size, img_size))
    base_arr = np.asarray(base, dtype="float32")

    # Upscale heatmap to image size (bilinear via PIL).
    hm_img = Image.fromarray((heatmap * 255).astype("uint8")).resize((img_size, img_size), Image.BILINEAR)
    hm = np.asarray(hm_img, dtype="float32") / 255.0

    color = _jet(hm).astype("float32")
    overlay = np.clip(base_arr * (1 - alpha) + color * alpha, 0, 255).astype("uint8")

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {
        "overlay_png_b64": b64,
        "focus_description": _describe_focus(heatmap),
        "available": True,
    }
