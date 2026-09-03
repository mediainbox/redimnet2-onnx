# Usage

Choose the API based on the input you already have:

| API                | Input                             | Best for                   |
|--------------------|-----------------------------------|----------------------------|
| `embed()`          | mono 16 kHz waveform              | most users                 |
| `embed_features()` | FP32 mel features `[1, 1, 72, T]` | custom preprocessing       |
| TensorRT runtime   | waveform or CUDA mel features     | optimized NVIDIA inference |

All embeddings are L2-normalized and shaped `[1, 192]`.

## Raw waveform

Install a runtime together with the `waveform` extra, then pass mono 16 kHz audio between 1 and 30 seconds:

```python
from redimnet2_onnx import load_model

model = load_model("b6-vb2+vox2_v0-lm")
embedding = model.embed(waveform)
```

`waveform` may be shaped `[samples]` or `[1, samples]`.

## Speaker verification

Compare two recordings using their embeddings:

```python
embedding_a = model.embed(waveform_a)[0]
embedding_b = model.embed(waveform_b)[0]

similarity = float(embedding_a @ embedding_b)
```

Because the embeddings are L2-normalized, the dot product is cosine similarity.

## Precomputed features

Use `embed_features()` when mel preprocessing is handled elsewhere:

```python
from redimnet2_onnx import load_model

model = load_model(
    "b6-vb2+vox2_v0-lm",
    providers=["CPUExecutionProvider"],
)

embedding = model.embed_features(features)
```

Input requirements:

- dtype: FP32
- shape: `[1, 1, 72, T]`
- `T` must be divisible by 4
- duration must be within the 1-30 second model profile

## Execution providers

Pass ONNX Runtime providers to `load_model()` in priority order:

```python
model = load_model(
    "b6-vb2+vox2_v0-lm",
    providers=[
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ],
)
```

Provider availability depends on the installed ONNX Runtime package and platform.

## Local ONNX files

`load_model()` is the normal high-level entry point: it downloads and verifies a released checkpoint.

Use `ReDimNet2` directly when the ONNX file is already managed locally:

```python
from redimnet2_onnx.runtime import ReDimNet2

model = ReDimNet2(
    "path/to/model.onnx",
    providers=["CPUExecutionProvider"],
)

embedding = model.embed_features(features)
```

## TensorRT

```python
from redimnet2_onnx.tensorrt import load_tensorrt_model

model = load_tensorrt_model(
    "b6-vb2+vox2_v0-lm",
    device_id=0,
)

embedding = model.embed(waveform_tensor)
```

The TensorRT path uses FP16 inference, GPU I/O binding, a coordinated CUDA stream, and persistent 1-, 10-, and 30-second
optimization-profile caches.

Engine caches are namespaced by model, runtime, TensorRT version, and GPU identity. Do not copy them between
incompatible environments.
