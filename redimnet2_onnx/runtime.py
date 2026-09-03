import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from redimnet2_onnx.models import download_model

Provider = str | tuple[str, dict[str, Any]]
MIN_FRAMES = 96
MAX_FRAMES = 2_996
TIME_ALIGNMENT = 4
OUTPUT_DIMENSION = 192


def _load_onnxruntime() -> Any:
    try:
        return importlib.import_module("onnxruntime")
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "ONNX Runtime is not installed; install a runtime extra such as 'redimnet2-onnx[cpu]' or "
            "'redimnet2-onnx[cuda]'"
        ) from error


def validate_session_contract(session: Any) -> tuple[str, str]:
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.name != "features" or list(model_input.shape[:3]) != [1, 1, 72]:
        raise RuntimeError(f"unexpected ONNX input contract: {model_input.name} {model_input.shape}")
    if model_output.name != "embedding" or list(model_output.shape) != [1, OUTPUT_DIMENSION]:
        raise RuntimeError(f"unexpected ONNX output contract: {model_output.name} {model_output.shape}")
    return model_input.name, model_output.name


class ReDimNet2:
    """Run a ReDimNet2 ONNX model with any ONNX Runtime provider."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Sequence[Provider] | None = None,
        session_options: Any | None = None,
    ) -> None:
        """Create an inference session for a local ONNX model.

        Args:
            model_path: Path to a ReDimNet2 ONNX model.
            providers: ONNX Runtime providers and optional provider settings.
            session_options: Optional ONNX Runtime session options.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ModuleNotFoundError: If ONNX Runtime is not installed.
            RuntimeError: If the model input or output contract is invalid.
        """
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        ort = _load_onnxruntime()
        self.model_path = path
        self.session = ort.InferenceSession(
            str(path), sess_options=session_options, providers=list(providers or ["CPUExecutionProvider"])
        )
        self.input_name, self.output_name = validate_session_contract(self.session)
        self._frontend: Any | None = None

    def embed_features(self, features: np.ndarray) -> np.ndarray:
        """Generate an embedding from mel features.

        Args:
            features: FP32 mel features shaped ``[1, 1, 72, T]``. ``T`` must be
                divisible by four and within the 1-30 second profile.

        Returns:
            An L2-normalized FP32 embedding shaped ``[1, 192]``.

        Raises:
            ValueError: If the input shape or time dimension is invalid.
            RuntimeError: If the model returns an invalid embedding.
        """
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 4 or tuple(values.shape[:3]) != (1, 1, 72):
            raise ValueError("features must have shape [1, 1, 72, T]")
        if not MIN_FRAMES <= values.shape[-1] <= MAX_FRAMES:
            raise ValueError(f"mel time must be within [{MIN_FRAMES}, {MAX_FRAMES}] frames")
        if values.shape[-1] % TIME_ALIGNMENT:
            raise ValueError("mel time must be divisible by 4")
        embedding = self.session.run([self.output_name], {self.input_name: np.ascontiguousarray(values)})[0]
        if embedding.shape != (1, OUTPUT_DIMENSION) or embedding.dtype != np.float32:
            raise RuntimeError(f"unexpected ONNX output: shape={embedding.shape}, dtype={embedding.dtype}")
        if not np.isfinite(embedding).all():
            raise RuntimeError("ONNX output contains non-finite values")
        return embedding

    def embed(self, waveforms: Any) -> np.ndarray:
        """Generate an embedding from a raw 16 kHz waveform.

        Args:
            waveforms: Mono audio shaped ``[samples]`` or ``[1, samples]``.

        Returns:
            An L2-normalized FP32 embedding shaped ``[1, 192]``.

        Raises:
            ModuleNotFoundError: If the optional ``waveform`` extra is not installed.
            ValueError: If the waveform duration or shape is invalid.
        """
        try:
            torch = importlib.import_module("torch")
            frontend_module = importlib.import_module("redimnet2_onnx.frontend")
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError("waveform embedding requires 'redimnet2-onnx[waveform]'") from error

        waveform_tensor = torch.as_tensor(waveforms)
        if self._frontend is None:
            self._frontend = frontend_module.ReDimNetFrontend().to(waveform_tensor.device).eval()
        else:
            self._frontend.to(waveform_tensor.device)
        features = self._frontend(waveform_tensor).cpu().numpy()
        return self.embed_features(features)


def load_model(
    model_name: str,
    *,
    model_cache_dir: str | Path | None = None,
    force_download: bool = False,
    providers: Sequence[Provider] | None = None,
    session_options: Any | None = None,
) -> ReDimNet2:
    """Download a model and create a provider-neutral runtime.

    Args:
        model_name: Public model identifier.
        model_cache_dir: Optional model download cache directory.
        force_download: Download the model again when true.
        providers: ONNX Runtime providers and optional provider settings.
        session_options: Optional ONNX Runtime session options.

    Returns:
        A ready-to-use ReDimNet2 runtime.
    """
    path = download_model(model_name, model_cache_dir, force_download=force_download)
    return ReDimNet2(path, providers=providers, session_options=session_options)
