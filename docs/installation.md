# Installation

Install one ONNX Runtime backend appropriate for the target platform.

| Platform         | Command                                  |
|------------------|------------------------------------------|
| CPU              | `pip install "redimnet2-onnx[cpu]"`      |
| NVIDIA CUDA      | `pip install "redimnet2-onnx[cuda]"`     |
| Intel OpenVINO   | `pip install "redimnet2-onnx[openvino]"` |
| Windows DirectML | `pip install "redimnet2-onnx[directml]"` |
| Qualcomm QNN     | `pip install "redimnet2-onnx[qnn]"`      |

Add the `waveform` extra when raw-waveform preprocessing is needed:

```bash
pip install "redimnet2-onnx[cuda,waveform]"
```

The same combination supports the optimized TensorRT runtime when compatible CUDA and TensorRT libraries are available
on the host.

!!! note "TensorRT version"

    Set `TENSORRT_VERSION` when the host does not expose TensorRT package metadata.

!!! warning "Runtime dependencies"

    Installing `redimnet2-onnx` without an extra installs neither ONNX Runtime nor PyTorch.
