# ReDimNet2 ONNX

[![CI status](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/ci.yaml/badge.svg)](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/ci.yaml)
[![Publish status](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/publish.yaml/badge.svg)](https://github.com/mediainbox/redimnet2-onnx/actions/workflows/publish.yaml)
[![image](https://img.shields.io/pypi/v/redimnet2-onnx.svg)](https://pypi.python.org/pypi/redimnet2-onnx)
[![image](https://img.shields.io/pypi/pyversions/redimnet2-onnx.svg)](https://pypi.python.org/pypi/redimnet2-onnx)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/pylint-dev/pylint)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![ONNX Model](https://img.shields.io/badge/model-ONNX-blue?logo=onnx&logoColor=white)](https://onnx.ai/)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://mediainbox.github.io/redimnet2-onnx/)
[![image](https://img.shields.io/pypi/l/redimnet2-onnx.svg)](https://pypi.python.org/pypi/redimnet2-onnx)

ONNX export of [PalabraAI ReDimNet2](https://github.com/PalabraAI/redimnet2) models. It produces normalized
192-dimensional speaker embeddings.

## Installation

By default, **no ONNX runtime is installed**. To run inference, you **must** install at least one ONNX backend using an
appropriate extra.

| Platform/Use Case  | Install Command                             | Notes                |
|--------------------|---------------------------------------------|----------------------|
| CPU (default)      | `pip install redimnet2-onnx[onnx]`          | Cross-platform       |
| NVIDIA GPU (CUDA)  | `pip install redimnet2-onnx[onnx-gpu]`      | Linux/Windows        |
| Intel (OpenVINO)   | `pip install redimnet2-onnx[onnx-openvino]` | Best on Intel CPUs   |
| Windows (DirectML) | `pip install redimnet2-onnx[onnx-directml]` | For DirectML support |
| Qualcomm (QNN)     | `pip install redimnet2-onnx[onnx-qnn]`      | Qualcomm chipsets    |
