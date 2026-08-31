# ReDimNet2 ONNX

ReDimNet2 ONNX provides ONNX exports of
[PalabraAI ReDimNet2](https://github.com/PalabraAI/redimnet2) speaker-recognition models. The models produce normalized,
192-dimensional speaker embeddings suitable for speaker verification and recognition.

## Runtime support

The package does not install an ONNX runtime by default. Choose the optional dependency that matches your hardware:

- ONNX Runtime for cross-platform CPU inference
- ONNX Runtime GPU for NVIDIA CUDA
- OpenVINO for Intel hardware
- DirectML for Windows accelerators
- QNN for supported Qualcomm devices

See [Installation](installation.md) for the available extras and install commands.

## Project links

- [Source code](https://github.com/mediainbox/redimnet2-onnx)
- [Package on PyPI](https://pypi.org/project/redimnet2-onnx/)
- [Issue tracker](https://github.com/mediainbox/redimnet2-onnx/issues)
