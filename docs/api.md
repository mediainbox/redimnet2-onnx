# API Reference

The API reference is generated from the package source.

For normal inference, start with `load_model()`. Use the runtime classes directly when the application manages ONNX
files itself.

## Models

::: redimnet2_onnx.models.available_models

::: redimnet2_onnx.models.get_model_spec

::: redimnet2_onnx.models.download_model

::: redimnet2_onnx.models.ModelSpec

## ONNX Runtime

::: redimnet2_onnx.runtime.load_model

::: redimnet2_onnx.runtime.ReDimNet2
    options:
      members:
        - embed_features
        - embed

## TensorRT

::: redimnet2_onnx.tensorrt.load_tensorrt_model

::: redimnet2_onnx.tensorrt.TensorRTReDimNet2
    options:
      members:
        - preprocess
        - embed_features
        - embed
