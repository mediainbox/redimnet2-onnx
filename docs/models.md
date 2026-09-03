# Models

ReDimNet2 ONNX includes all 20 published ReDimNet2 checkpoints.

Model identifiers follow:

```text
<architecture>-<dataset>-<training>
```

For example:

```text
b6-vb2+vox2_v0-lm
```

## Architecture

Architectures range from `b0` through `b6`. Larger variants increase model capacity and compute.

## Training variants

- `lm` - large-margin fine-tuned weights; recommended upstream for inference.
- `ptn` - pretrained weights before large-margin fine-tuning.

## Released families

| Architecture | Training data                     | Variants           |
|--------------|-----------------------------------|--------------------|
| B0-B6        | VoxCeleb2                         | `ptn`, `lm`        |
| B6           | VoxBlink2 + VoxCeleb2             | `ptn`, `lm`        |
| B3, B6       | VoxBlink2 + VoxCeleb2 + CN-Celeb2 | `ptn`, `lm`        |

The examples in these docs use `b6-vb2+vox2_v0-lm`. Select the checkpoint and verification threshold using validation
data representative of your application.

## Discover models

Use the installed package as the source of truth for exact model identifiers:

```python
from redimnet2_onnx import available_models

for name in available_models():
    print(name)
```

Inspect metadata for one checkpoint with:

```python
from redimnet2_onnx.models import get_model_spec

spec = get_model_spec("b6-vb2+vox2_v0-lm")

print(spec.model_name)
print(spec.dataset)
print(spec.train_type)
print(spec.source_url)
```

For upstream benchmark results and the original PyTorch checkpoints, see the
[official ReDimNet2 repository](https://github.com/PalabraAI/redimnet2).

## Validate release artifacts

Maintainers can compare every exported model with the pinned PyTorch source and TensorRT FP16:

```bash
uv run python scripts/validate_tensorrt.py \
  artifacts/models-v1 \
  /path/to/voice-benchmark \
  --max-speakers 0 \
  --max-recordings-per-speaker 0 \
  --resume
```

The audio directory must contain one folder per speaker with at least two 16 kHz recordings. Zero limits use all
qualifying data. Set `TENSORRT_VERSION` when it cannot be detected from the installed packages or host environment.
