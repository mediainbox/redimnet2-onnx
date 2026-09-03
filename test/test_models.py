import hashlib
import json
from pathlib import Path

import pytest

from redimnet2_onnx.models import available_models, download_model, get_model_spec

TEST_MODEL = "b6-vb2+vox2_v0-lm"


def test_official_model_registry() -> None:
    names = available_models()
    assert len(names) == 20
    assert "b0-vox2-ptn" in names
    assert TEST_MODEL in names
    assert "b6-vb2+vox2+cnc2_v0-lm" in names


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_model_spec("missing")


def test_download_uses_manifest_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = get_model_spec(TEST_MODEL)
    release = tmp_path / "release"
    release.mkdir()
    contents = b"test ONNX artifact"
    (release / spec.filename).write_bytes(contents)
    manifest = {
        "schema_version": 1,
        "models": {
            spec.name: {
                "filename": spec.filename,
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        },
    }
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("REDIMNET2_ONNX_RELEASE_BASE_URL", release.as_uri())

    model_path = download_model(spec.name, tmp_path / "cache")

    assert model_path.read_bytes() == contents
