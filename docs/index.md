# ReDimNet2 ONNX

Run the official [PalabraAI ReDimNet2](https://github.com/PalabraAI/redimnet2) speaker-verification models with ONNX
Runtime.

All 20 published checkpoints are available and produce L2-normalized 192-dimensional speaker embeddings.

## Quick start

For raw 16 kHz audio on CPU:

```bash
pip install "redimnet2-onnx[cpu,waveform]"
```

```python
from redimnet2_onnx import load_model

model = load_model("b6-vb2+vox2_v0-lm")

# waveform: mono 16 kHz audio, 1-30 seconds
embedding = model.embed(waveform)

print(embedding.shape)
# (1, 192)
```

Models are downloaded from versioned GitHub Releases and verified with SHA-256.

## Next steps

- [Installation](installation.md) - choose an inference backend.
- [Usage](usage.md) - waveform inference, speaker verification, providers, and TensorRT.
- [Models](models.md) - understand model names and released checkpoint families.
- [API Reference](api.md) - complete Python API.
