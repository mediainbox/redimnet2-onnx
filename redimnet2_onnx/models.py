import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final
from urllib.parse import quote

GITHUB_REPOSITORY: Final = "mediainbox/redimnet2-onnx"
MODEL_RELEASE: Final = "models-v1"
UPSTREAM_REPOSITORY: Final = "PalabraAI/redimnet2"
UPSTREAM_COMMIT: Final = "c5bbe0b76e37df698c403f8844e41304ceab6307"
UPSTREAM_WEIGHT_RELEASE: Final = "v1.0.0"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Describe a released ReDimNet2 model.

    Attributes:
        name: Public model identifier used by the package.
        model_name: Upstream ReDimNet2 architecture name.
        dataset: Upstream training dataset identifier.
        train_type: Upstream training type.
    """

    name: str
    model_name: str
    dataset: str
    train_type: str

    @property
    def filename(self) -> str:
        """Return the ONNX release filename."""
        return f"{self.name}.onnx"

    @property
    def source_filename(self) -> str:
        """Return the upstream PyTorch checkpoint filename."""
        return f"{self.model_name}-{self.dataset}-{self.train_type}.pt"

    @property
    def source_url(self) -> str:
        """Return the upstream PyTorch checkpoint URL."""
        return (
            f"https://github.com/{UPSTREAM_REPOSITORY}/releases/download/"
            f"{UPSTREAM_WEIGHT_RELEASE}/{quote(self.source_filename, safe='+._-')}"
        )


_MODEL_SPECS = (
    ModelSpec("b0-vox2-ptn", "b0", "vox2", "ptn"),
    ModelSpec("b0-vox2-lm", "b0", "vox2", "lm"),
    ModelSpec("b1-vox2-ptn", "b1", "vox2", "ptn"),
    ModelSpec("b1-vox2-lm", "b1", "vox2", "lm"),
    ModelSpec("b2-vox2-ptn", "b2", "vox2", "ptn"),
    ModelSpec("b2-vox2-lm", "b2", "vox2", "lm"),
    ModelSpec("b3-vox2-ptn", "b3", "vox2", "ptn"),
    ModelSpec("b3-vox2-lm", "b3", "vox2", "lm"),
    ModelSpec("b4-vox2-ptn", "b4", "vox2", "ptn"),
    ModelSpec("b4-vox2-lm", "b4", "vox2", "lm"),
    ModelSpec("b5-vox2-ptn", "b5", "vox2", "ptn"),
    ModelSpec("b5-vox2-lm", "b5", "vox2", "lm"),
    ModelSpec("b6-vox2-ptn", "b6", "vox2", "ptn"),
    ModelSpec("b6-vox2-lm", "b6", "vox2", "lm"),
    ModelSpec("b6-vb2+vox2_v0-ptn", "b6", "vb2+vox2_v0", "ptn"),
    ModelSpec("b6-vb2+vox2_v0-lm", "b6", "vb2+vox2_v0", "lm"),
    ModelSpec("b3-vb2+vox2+cnc2_v0-ptn", "b3", "vb2+vox2+cnc2_v0", "ptn"),
    ModelSpec("b3-vb2+vox2+cnc2_v0-lm", "b3", "vb2+vox2+cnc2_v0", "lm"),
    ModelSpec("b6-vb2+vox2+cnc2_v0-ptn", "b6", "vb2+vox2+cnc2_v0", "ptn"),
    ModelSpec("b6-vb2+vox2+cnc2_v0-lm", "b6", "vb2+vox2+cnc2_v0", "lm"),
)

MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType({spec.name: spec for spec in _MODEL_SPECS})


def available_models() -> tuple[str, ...]:
    """Return all supported model identifiers."""
    return tuple(MODEL_SPECS)


def get_model_spec(model_name: str) -> ModelSpec:
    """Return metadata for a supported model.

    Args:
        model_name: Public model identifier.

    Returns:
        Metadata for the requested model.

    Raises:
        ValueError: If the model identifier is unknown.
    """
    try:
        return MODEL_SPECS[model_name]
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(f"unknown model {model_name!r}; available models: {choices}") from error


def _default_cache_dir() -> Path:
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home / "redimnet2-onnx"


def _release_base_url() -> str:
    override = os.environ.get("REDIMNET2_ONNX_RELEASE_BASE_URL")
    if override:
        return override.rstrip("/")
    return f"https://github.com/{GITHUB_REPOSITORY}/releases/download/{MODEL_RELEASE}"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(destination)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {error}") from error


def _load_release_manifest(cache_dir: Path, force_download: bool) -> dict[str, Any]:
    manifest_path = cache_dir / MODEL_RELEASE / "manifest.json"
    if force_download or not manifest_path.is_file():
        _download(f"{_release_base_url()}/manifest.json", manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid model release manifest at {manifest_path}: {error}") from error
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
        raise RuntimeError(f"unsupported model release manifest at {manifest_path}")
    return manifest


def download_model(
    model_name: str,
    cache_dir: str | Path | None = None,
    *,
    force_download: bool = False,
) -> Path:
    """Download and verify an ONNX model from the GitHub release.

    Args:
        model_name: Public model identifier.
        cache_dir: Optional model cache directory.
        force_download: Download the manifest and model again when true.

    Returns:
        Path to the verified ONNX model.

    Raises:
        RuntimeError: If the manifest is invalid or checksum verification fails.
        ValueError: If the model identifier is unknown.
    """
    spec = get_model_spec(model_name)
    cache_root = Path(cache_dir or _default_cache_dir()).expanduser().resolve()
    manifest = _load_release_manifest(cache_root, force_download)
    entry = manifest["models"].get(spec.name)
    if not isinstance(entry, dict):
        raise RuntimeError(f"model {spec.name!r} is not present in release {MODEL_RELEASE}")
    if entry.get("filename") != spec.filename:
        raise RuntimeError(f"release manifest filename mismatch for {spec.name}")
    expected_sha = entry.get("sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise RuntimeError(f"release manifest has no valid SHA-256 for {spec.name}")

    destination = cache_root / MODEL_RELEASE / spec.filename
    if destination.is_file() and not force_download and file_sha256(destination) == expected_sha:
        return destination
    _download(f"{_release_base_url()}/{quote(spec.filename, safe='+._-')}", destination)
    actual_sha = file_sha256(destination)
    if actual_sha != expected_sha:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"SHA-256 mismatch for {spec.name}: expected {expected_sha}, got {actual_sha}")
    return destination
