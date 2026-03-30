'''from huggingface_hub import snapshot_download
import os

base_path = "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/models"

print("Downloading SmolLM2-1.7B (Optimized)...")
snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    local_dir=os.path.join(base_path, "smollm2-1.7b"),
    # Only download the essential files for your Python script
    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt", "*.py"],
    ignore_patterns=["*.gguf", "*.onnx", "*-q4_k_m.ps"], 
)

print("✅ Successfully downloaded to models/smollm2-1.7b")'''

# Downloading Qwen2.5-1.5B-Instruct (Optimized) from Hugging Face Hub
from huggingface_hub import snapshot_download
import os

base_path = "/Users/sudiptogoldfish/Documents/BenchJudge A Meta Evaluation Framework for Auditing LLM as a Judge Systems/models"

print("Downloading Qwen2.5-1.5B-Instruct...")
snapshot_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    local_dir=os.path.join(base_path, "qwen2.5-1.5b"),
    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
    local_dir_use_symlinks=False
)

print("✅ Successfully downloaded Qwen to models/qwen2.5-1.5b")