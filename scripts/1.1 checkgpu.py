import torch
if torch.backends.mps.is_available():
    print("✅ Metal GPU (MPS) is available!")
else:
    print("❌ MPS not found. Check your macOS version (requires 12.3+).")