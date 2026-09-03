# ReDimNet2 ONNX

[![CI status](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/ci.yaml/badge.svg)](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/ci.yaml)
[![Publish status](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/publish.yaml/badge.svg)](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/publish.yaml)
[![PyPI](https://img.shields.io/pypi/v/redimnet2-onnx.svg)](https://pypi.org/project/redimnet2-onnx/)
[![Python](https://img.shields.io/pypi/pyversions/redimnet2-onnx.svg)](https://pypi.org/project/redimnet2-onnx/)
[![License](https://img.shields.io/pypi/l/redimnet2-onnx.svg)](https://pypi.org/project/redimnet2-onnx/)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://mediainbox.github.io/redimnet2-onnx/)
[![ONNX Model](https://img.shields.io/badge/model-ONNX-blue?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Run the official [PalabraAI ReDimNet2](https://github.com/PalabraAI/redimnet2) speaker-verification models with
ONNX Runtime. All 20 published checkpoints are available and produce L2-normalized 192-dimensional embeddings.

<!-- TOC -->
* [ReDimNet2 ONNX](#redimnet2-onnx)
  * [Installation](#installation)
  * [Quick Start](#quick-start)
  * [Models](#models)
  * [Mel Features](#mel-features)
  * [TensorRT](#tensorrt)
<!-- TOC -->

## Installation

For raw 16 kHz audio on CPU:

```bash
pip install "redimnet2-onnx[cpu,waveform]"
```

Other supported setups:

| Use case                      | Installation                                  |
|-------------------------------|-----------------------------------------------|
| Mel features on CPU           | `pip install "redimnet2-onnx[cpu]"`           |
| Mel features on NVIDIA CUDA   | `pip install "redimnet2-onnx[cuda]"`          |
| Raw waveform on CPU           | `pip install "redimnet2-onnx[cpu,waveform]"`  |
| Raw waveform on CUDA/TensorRT | `pip install "redimnet2-onnx[cuda,waveform]"` |
| Intel OpenVINO                | `pip install "redimnet2-onnx[openvino]"`      |
| Windows DirectML              | `pip install "redimnet2-onnx[directml]"`      |
| Qualcomm QNN                  | `pip install "redimnet2-onnx[qnn]"`           |

## Quick Start

Pass a mono 16 kHz waveform as a NumPy array or PyTorch tensor. Audio must be between 1 and 30 seconds.

```python
from redimnet2_onnx import load_model

model = load_model("b6-vb2+vox2_v0-lm")
embedding = model.embed(waveform)

print(embedding.shape)  # (1, 192)
```

Models are downloaded once from the versioned GitHub Release, cached locally, and verified with SHA-256.

To compare two voices, use cosine similarity. The embeddings are already normalized:

```python
similarity = (embedding_a @ embedding_b.T).item()
```

## Models

List the available checkpoints:

```python
from redimnet2_onnx import available_models, load_model

print(available_models())
model = load_model("b6-vb2+vox2_v0-lm")
```

## Mel Features

PyTorch is optional when mel features are produced elsewhere. The ONNX input is FP32 `[1, 1, 72, T]`, where `T`
must be divisible by four and represent approximately 1-30 seconds of audio.

```python
from redimnet2_onnx import load_model

model = load_model("b6-vb2+vox2_v0-lm", providers=["CUDAExecutionProvider"])
embedding = model.embed_features(features)
```

## TensorRT

For ONNX Runtime TensorRT EP with FP16, CUDA I/O binding, and persistent engine caches:

```python
from redimnet2_onnx.tensorrt import load_tensorrt_model

model = load_tensorrt_model("b6-vb2+vox2_v0-lm")
embedding = model.embed(waveform_tensor)
```

This requires compatible CUDA and TensorRT libraries on the host. The TensorRT Python package is not required.

See the [documentation](https://mediainbox.github.io/redimnet2-onnx/) for the complete API reference.
