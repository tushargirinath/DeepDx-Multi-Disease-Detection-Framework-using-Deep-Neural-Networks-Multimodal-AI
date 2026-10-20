"""
Export the EXACT ordered class-name list your model was trained on.

WHY THIS MATTERS
----------------
Your CNN's output neuron #i corresponds to the i-th label in
`sorted(df['label'].unique())`. The web app maps prediction indices back to
disease names using `class_names.json`. If that file's order does not match
your training order, every prediction will show the WRONG disease name even
though the model is correct.

The `class_names.json` shipped with this app is a best-effort RECONSTRUCTION.
You should overwrite it with the authoritative list from your own data.

HOW TO USE
----------
Paste this snippet into a cell in your training notebook / Colab, where the
`df` DataFrame (with a 'label' column) still exists, then download the file
it writes and drop it into the project root, replacing class_names.json.
"""

import json

# `df` is the DataFrame you built in Cell 3 of the training notebook.
CLASS_NAMES = sorted(df["label"].unique().tolist())  # noqa: F821

assert len(CLASS_NAMES) == 30, f"Expected 30 classes, got {len(CLASS_NAMES)}"

with open("class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)

print(f"Wrote class_names.json with {len(CLASS_NAMES)} classes:")
for i, c in enumerate(CLASS_NAMES):
    print(f"{i:>2}: {c}")

# In Colab, download it with:
#   from google.colab import files; files.download("class_names.json")
