import hashlib
import json
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # pylint: disable=import-error
import torch  # pylint: disable=import-error

from redimnet2_onnx.frontend import ReDimNetFrontend
from redimnet2_onnx.models import download_model
from redimnet2_onnx.runtime import MAX_FRAMES, MIN_FRAMES, OUTPUT_DIMENSION, TIME_ALIGNMENT, validate_session_contract


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensorrt_version() -> str:
    for variable in ("TENSORRT_VERSION", "NVIDIA_TENSORRT_VERSION"):
        if value := os.environ.get(variable):
            return value
    try:
        return version("tensorrt")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "could not determine the TensorRT runtime version; set TENSORRT_VERSION when TensorRT is supplied "
            "by the host or container"
        ) from error


def _safe_component(value: str) -> str:
    return "".join(character if character.isalnum() or character in ".-_" else "_" for character in value)


def _cache_namespace(model_path: Path, device_id: int) -> str:
    properties = torch.cuda.get_device_properties(device_id)
    identity = {
        "model_sha256": _file_sha256(model_path),
        "onnxruntime": ort.__version__,
        "tensorrt": _tensorrt_version(),
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return (
        f"onnx-{identity['model_sha256'][:12]}-ort-{_safe_component(ort.__version__)}"
        f"-trt-{_safe_component(identity['tensorrt'])}-sm{properties.major}{properties.minor}-{identity_hash}"
    )


def _default_engine_cache_dir() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "redimnet2-onnx" / "tensorrt"


class TensorRTReDimNet2:
    """Run ReDimNet2 with FP32 preprocessing/output normalization and a TensorRT FP16 backend."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        engine_cache_dir: str | Path | None = None,
        device_id: int = 0,
    ) -> None:
        """Create a TensorRT runtime for a local ONNX model.

        Args:
            model_path: Path to a ReDimNet2 ONNX model.
            engine_cache_dir: Optional TensorRT engine cache directory.
            device_id: CUDA device index.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If CUDA or the required ONNX Runtime providers are unavailable.
        """
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT inference requires CUDA")
        required = {"TensorrtExecutionProvider", "CUDAExecutionProvider"}
        available = set(ort.get_available_providers())
        if missing := sorted(required - available):
            raise RuntimeError(f"required providers unavailable: {missing}; available: {sorted(available)}")

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        self.model_path = path
        self.device = torch.device(f"cuda:{device_id}")
        cache_root = Path(engine_cache_dir or _default_engine_cache_dir()).expanduser().resolve()
        self.engine_cache_dir = cache_root / _cache_namespace(path, device_id)
        self.engine_cache_dir.mkdir(parents=True, exist_ok=True)
        self.frontend = ReDimNetFrontend().to(self.device).eval()
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        stream_pointer = str(self.stream.cuda_stream)

        def shape(frames: int) -> str:
            return f"features:1x1x72x{frames}"

        trt_options: dict[str, Any] = {
            "device_id": device_id,
            "user_compute_stream": stream_pointer,
            "trt_fp16_enable": True,
            "trt_max_workspace_size": 4 * 1024**3,
            "trt_profile_min_shapes": shape(MIN_FRAMES),
            "trt_profile_opt_shapes": shape(996),
            "trt_profile_max_shapes": shape(MAX_FRAMES),
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(self.engine_cache_dir),
            "trt_engine_cache_prefix": "engine",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(self.engine_cache_dir),
        }
        cuda_options: dict[str, Any] = {"user_compute_stream": stream_pointer}
        if device_id:
            cuda_options["device_id"] = device_id
        self.session = ort.InferenceSession(
            str(path),
            providers=[
                ("TensorrtExecutionProvider", trt_options),
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ],
        )
        if self.session.get_providers()[0] != "TensorrtExecutionProvider":
            raise RuntimeError(f"TensorRT EP is not primary: {self.session.get_providers()}")
        self.input_name, self.output_name = validate_session_contract(self.session)

    def preprocess(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Convert a raw 16 kHz waveform to aligned FP32 mel features.

        Args:
            waveforms: Mono audio shaped ``[samples]`` or ``[1, samples]``.

        Returns:
            FP32 mel features on the configured CUDA device.
        """
        return self.frontend(waveforms.to(self.device, dtype=torch.float32, non_blocking=True))

    def embed_features(self, features: torch.Tensor) -> torch.Tensor:
        """Generate an embedding from CUDA mel features.

        Args:
            features: FP32 CUDA mel features shaped ``[1, 1, 72, T]``.

        Returns:
            An L2-normalized FP32 CUDA embedding shaped ``[1, 192]``.

        Raises:
            ValueError: If the input device, dtype, or shape is invalid.
        """
        if features.device != self.device or features.dtype != torch.float32:
            raise ValueError(f"features must be FP32 on {self.device}")
        if features.ndim != 4 or tuple(features.shape[:3]) != (1, 1, 72):
            raise ValueError("features must have shape [1, 1, 72, T]")
        if not MIN_FRAMES <= features.shape[-1] <= MAX_FRAMES:
            raise ValueError(f"mel time must be within [{MIN_FRAMES}, {MAX_FRAMES}] frames")
        if features.shape[-1] % TIME_ALIGNMENT:
            raise ValueError("mel time must be divisible by 4")

        features = features.contiguous()
        output = torch.empty((1, OUTPUT_DIMENSION), dtype=torch.float32, device=self.device)
        caller_stream = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            binding = self.session.io_binding()
            binding.bind_input(
                self.input_name,
                "cuda",
                self.device.index or 0,
                np.float32,
                tuple(features.shape),
                features.data_ptr(),
            )
            binding.bind_output(
                self.output_name,
                "cuda",
                self.device.index or 0,
                np.float32,
                tuple(output.shape),
                output.data_ptr(),
            )
            self.session.run_with_iobinding(binding)
        caller_stream.wait_stream(self.stream)
        return torch.nn.functional.normalize(output, dim=1)

    def embed(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Generate an embedding from a raw 16 kHz waveform.

        Args:
            waveforms: Mono audio shaped ``[samples]`` or ``[1, samples]``.

        Returns:
            An L2-normalized FP32 CUDA embedding shaped ``[1, 192]``.
        """
        return self.embed_features(self.preprocess(waveforms))


def load_tensorrt_model(
    model_name: str,
    *,
    model_cache_dir: str | Path | None = None,
    engine_cache_dir: str | Path | None = None,
    force_download: bool = False,
    device_id: int = 0,
) -> TensorRTReDimNet2:
    """Download a model and create a TensorRT runtime.

    Args:
        model_name: Public model identifier.
        model_cache_dir: Optional model download cache directory.
        engine_cache_dir: Optional TensorRT engine cache directory.
        force_download: Download the model again when true.
        device_id: CUDA device index.

    Returns:
        A ready-to-use TensorRT runtime.
    """
    path = download_model(model_name, model_cache_dir, force_download=force_download)
    return TensorRTReDimNet2(path, engine_cache_dir=engine_cache_dir, device_id=device_id)
