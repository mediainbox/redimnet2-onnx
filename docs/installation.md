# Installation

ReDimNet2 ONNX requires Python 3.12 or newer. Install one runtime extra appropriate for the target platform.

| Platform or use case | Command                                       | Notes                        |
|----------------------|-----------------------------------------------|------------------------------|
| CPU                  | `pip install "redimnet2-onnx[onnx]"`          | Cross-platform               |
| NVIDIA GPU           | `pip install "redimnet2-onnx[onnx-gpu]"`      | CUDA on Linux or Windows     |
| Intel                | `pip install "redimnet2-onnx[onnx-openvino]"` | Optimized for Intel hardware |
| Windows accelerators | `pip install "redimnet2-onnx[onnx-directml]"` | DirectML support             |
| Qualcomm             | `pip install "redimnet2-onnx[onnx-qnn]"`      | Supported Qualcomm devices   |

!!! note

    Installing `redimnet2-onnx` without an extra does not install an inference runtime.

## Development installation

Clone the repository and synchronize the locked development environment with [uv](https://docs.astral.sh/uv/):

```console
git clone https://github.com/mediainbox/redimnet2-onnx.git
cd redimnet2-onnx
uv sync --locked --all-groups
```

Build and preview the documentation locally:

```console
uv run mkdocs serve
```
