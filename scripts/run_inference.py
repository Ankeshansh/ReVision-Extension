"""
Run inference using ReVision model.

Example:
    python scripts/run_inference.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from PIL import Image

# IMPORTANT: use same imports as finetune
from src.model.processor import ReVisionProcessor
from src.model.revision_model import ReVisionForConditionalGeneration

# SAME MODEL_ID as finetune.py
MODEL_ID = "anonymoususerrevision/ReVision-250M-256-16"


def main():
    
    use_auth_token = os.getenv("HF_TOKEN") 

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---- LOAD MODEL + PROCESSOR ----
    print("Loading processor...")
    processor = ReVisionProcessor.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )

    print("Loading model...")
    model = ReVisionForConditionalGeneration.from_pretrained(
        MODEL_ID, use_auth_token=use_auth_token
    )

    model.to(device)
    model.eval()

    # ---- SAMPLE INPUT ----
    prompt = "Rewrite the following question clearly and concisely: How can I store cookie butter?"

    # Dummy image (model expects image input)
    image_path = "sample.jpg"  # or any default image

    if os.path.exists(image_path):
        image = Image.open(image_path).convert("RGB")
    else:
        print("No image found, using blank image")
        image = Image.new("RGB", (256, 256), color="white")

    # ---- PREPROCESS ----
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]

    # ---- INFERENCE ----
    print("Running inference...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False
        )

    # ---- POSTPROCESS ----
    generated_tokens = outputs[0][input_len:]
    decoded = processor.decode(generated_tokens, skip_special_tokens=True)

    # clean chat artifacts
    decoded = decoded.replace("assistant", "").strip()

    # ---- RESULT ----
    print("\n--- RESULT ---")
    print("Input Prompt:", prompt)
    print("Rewritten:", decoded)


if __name__ == "__main__":
    main()
