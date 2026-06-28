"""Redirect HuggingFace caches inside the build venv so model downloads are part of the build
environment, not the host home dir (~/.cache/huggingface). Import this FIRST — before any
huggingface_hub/transformers import — since those read HF_HOME at import time."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
HF_HOME = os.path.join(_HERE, ".venv", "hf_cache")  # hub cache lands at $HF_HOME/hub
os.environ["HF_HOME"] = HF_HOME
