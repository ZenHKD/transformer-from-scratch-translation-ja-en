#!/bin/bash

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
uv pip install huggingface_hub numpy pandas datasets tokenizers torchmetrics tqdm tensorboard
