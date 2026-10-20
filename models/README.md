# DeepDx Model Weights

This folder contains the trained CNN model weights required for real inference.

### Model

```text
models/best_model.weights.h5
```

The `best_model.weights.h5` file is tracked and stored using **Git LFS** because of its large size.

* **Model file:** `best_model.weights.h5`
* **Storage:** Git LFS
* **Purpose:** CNN-based disease image classification and inference
* If the model weights are unavailable, the application can still start in **DEMO MODE** with simulated predictions, clearly indicated in the UI.
* The model path can be customized through `CNN_WEIGHTS_PATH` in your `.env` file.

The application reconstructs the required CNN architecture and loads the trained weights using `model.load_weights(...)`, so the full `.keras` model file is not required.
