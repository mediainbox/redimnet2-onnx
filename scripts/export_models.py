import argparse
import hashlib
import importlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch  # pylint: disable=import-error

from redimnet2_onnx.models import (
    MODEL_RELEASE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    ModelSpec,
    available_models,
    file_sha256,
    get_model_spec,
)

VALIDATION_FRAMES = (496, 996, 2_996)
UPSTREAM_HUB_REF = f"{UPSTREAM_REPOSITORY}:{UPSTREAM_COMMIT}"
ONNX_REFERENCE_MINIMUM_COSINE = 0.99999
ONNX_REFERENCE_MAXIMUM_SCORE_DRIFT_P99 = 5e-4
TENSORRT_REFERENCE_MINIMUM_COSINE = 0.9999
TENSORRT_REFERENCE_MAXIMUM_SCORE_DRIFT_P99 = 2e-3
REFERENCE_MINIMUM_TOP1_AGREEMENT = 0.995
REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP = 0.005


@dataclass(frozen=True)
class ReferenceAudio:
    waveforms: tuple[torch.Tensor, ...]
    speaker_ids: tuple[str, ...]


class ExternallyAlignedReDimNet(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module) -> None:
        super().__init__()
        if backbone.is_subnet:
            raise ValueError("subnet-mode ReDimNet2 is not supported")
        self.backbone = backbone

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        network = self.backbone
        output = network.stem(features)
        if network.agg_gnorm:
            output = network.stem_gnorm(output)
        outputs_1d = [output]
        for stage_index in range(network.num_stages):
            outputs_1d.extend(network.run_stage(outputs_1d, stage_index))
        output = network.fin_wght1d(outputs_1d)
        output = network.fin_to2d(output)
        return network.head(output)


class DynamicAstp(torch.nn.Module):
    def __init__(self, pool: torch.nn.Module) -> None:
        super().__init__()
        self.global_context_att = pool.global_context_att
        self.linear1 = pool.linear1
        self.linear2 = pool.linear2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            batch, channels, frequencies, frames = features.shape
            features = features.reshape(batch, channels * frequencies, frames)
        if self.global_context_att:
            mean = torch.mean(features, dim=-1, keepdim=True)
            centered = features - mean
            variance = torch.sum(centered * centered, dim=-1, keepdim=True) / (features.shape[-1] - 1)
            std = torch.sqrt(variance + 1e-7)
            attention_input = torch.cat((features, mean.expand_as(features), std.expand_as(features)), dim=1)
        else:
            attention_input = features
        alpha = torch.tanh(self.linear1(attention_input))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * features, dim=2)
        variance = torch.sum(alpha * features.square(), dim=2) - mean.square()
        std = torch.sqrt(variance.clamp(min=1e-7))
        return torch.cat((mean, std), dim=1)


class NormalizedMelEmbedding(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.backbone = ExternallyAlignedReDimNet(model.backbone)
        self.pool = DynamicAstp(model.pool)
        self.bn = model.bn
        self.linear = model.linear
        self.bn2 = model.bn2
        self.before_pool_offset = model.before_pool_offset

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.backbone(features)
        if output.ndim == 4:
            batch, channels, frequencies, frames = output.shape
            output = output.reshape(batch, channels * frequencies, frames)
        if self.before_pool_offset is not None:
            output = output[:, :, self.before_pool_offset :]
        output = self.bn(self.pool(output))
        output = self.linear(output)
        if self.bn2 is not None:
            output = self.bn2(output)
        return torch.nn.functional.normalize(output.float(), dim=1)


def _load_upstream_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    model = torch.hub.load(
        UPSTREAM_HUB_REF,
        "redimnet2",
        model_name=spec.model_name,
        train_type=spec.train_type,
        dataset=spec.dataset,
        pretrained=True,
        trust_repo=True,
        skip_validation=True,
    )
    return model.to(device).eval()


def _dynamic_shapes() -> dict[str, dict[int, Any]]:
    frame_groups = torch.export.Dim("mel_frame_groups", min=24, max=749)
    return {"features": {3: 4 * frame_groups}}


def _contract(path: Path) -> dict[str, Any]:
    onnx = importlib.import_module("onnx")

    model = onnx.load(path, load_external_data=False)
    onnx.checker.check_model(model, full_check=True)
    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    input_shape = [dimension.dim_value or dimension.dim_param for dimension in graph_input.type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value or dimension.dim_param for dimension in graph_output.type.tensor_type.shape.dim]
    if graph_input.name != "features" or input_shape[:3] != [1, 1, 72] or not isinstance(input_shape[3], str):
        raise RuntimeError(f"invalid ONNX input contract: {graph_input.name} {input_shape}")
    if graph_output.name != "embedding" or output_shape != [1, 192]:
        raise RuntimeError(f"invalid ONNX output contract: {graph_output.name} {output_shape}")
    return {
        "input_name": graph_input.name,
        "input_shape": input_shape,
        "output_name": graph_output.name,
        "output_shape": output_shape,
    }


def _true_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def _pairwise_scores(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    matrix = normalized @ normalized.T
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def _nearest_neighbors(embeddings: np.ndarray) -> np.ndarray:
    normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    scores = normalized @ normalized.T - np.eye(len(normalized), dtype=np.float32) * 2
    return np.argmax(scores, axis=1)


def _numerical_report(  # pylint: disable=too-many-locals
    expected: np.ndarray, actual: np.ndarray, speaker_ids: tuple[str, ...] | None = None
) -> dict[str, Any]:
    if expected.shape != actual.shape or expected.ndim != 2 or expected.shape[1] != 192:
        raise RuntimeError(f"invalid embedding shapes: expected={expected.shape}, actual={actual.shape}")
    if not np.isfinite(actual).all():
        raise RuntimeError("ONNX output contains non-finite values")
    expected_scores = _pairwise_scores(expected)
    actual_scores = _pairwise_scores(actual)
    score_drift = np.abs(expected_scores - actual_scores)
    thresholds = np.asarray([0.2, 0.32, 0.5], dtype=np.float32)
    threshold_flip_mask = (expected_scores[:, None] >= thresholds) != (actual_scores[:, None] >= thresholds)
    threshold_flips = np.count_nonzero(threshold_flip_mask)
    flipped_reference_margins = np.abs(expected_scores[:, None] - thresholds)[threshold_flip_mask]
    expected_neighbors = _nearest_neighbors(expected)
    actual_neighbors = _nearest_neighbors(actual)
    norms = np.linalg.norm(actual, axis=1)
    report = {
        "minimum_true_cosine": float(_true_cosine(expected, actual).min()),
        "pairwise_score_drift_p99": float(np.quantile(score_drift, 0.99)),
        "pairwise_score_drift_maximum": float(score_drift.max()),
        "threshold_flips": int(threshold_flips),
        "threshold_flip_rate": float(threshold_flips / threshold_flip_mask.size),
        "threshold_flip_counts": {
            f"{threshold:g}": int(threshold_flip_mask[:, index].sum()) for index, threshold in enumerate(thresholds)
        },
        "maximum_threshold_flip_reference_margin": (
            float(flipped_reference_margins.max()) if flipped_reference_margins.size else 0.0
        ),
        "top1_index_agreement": float(np.mean(expected_neighbors == actual_neighbors)),
        "minimum_output_norm": float(norms.min()),
        "maximum_output_norm": float(norms.max()),
    }
    if speaker_ids is None:
        report["top1_agreement"] = report["top1_index_agreement"]
        return report
    if len(speaker_ids) != len(actual):
        raise RuntimeError(f"speaker label count does not match embeddings: {len(speaker_ids)} != {len(actual)}")
    people = np.asarray(speaker_ids)
    expected_people = people[expected_neighbors]
    actual_people = people[actual_neighbors]
    pair_indices = np.triu_indices(len(people), k=1)
    genuine_mask = people[pair_indices[0]] == people[pair_indices[1]]
    report.update(
        {
            "genuine_pairwise_score_drift_p99": float(np.quantile(score_drift[genuine_mask], 0.99)),
            "impostor_pairwise_score_drift_p99": float(np.quantile(score_drift[~genuine_mask], 0.99)),
            "top1_agreement": float(np.mean(expected_people == actual_people)),
            "reference_top1_speaker_accuracy": float(np.mean(expected_people == people)),
            "candidate_top1_speaker_accuracy": float(np.mean(actual_people == people)),
        }
    )
    return report


def _require_parity(
    report: dict[str, Any],
    minimum_cosine: float,
    maximum_score_drift: float,
    backend: str,
    *,
    minimum_top1_agreement: float = 1.0,
    maximum_top1_accuracy_drop: float = 0.0,
) -> None:
    if report["minimum_true_cosine"] < minimum_cosine:
        raise RuntimeError(f"{backend} cosine validation failed: {report}")
    if report["pairwise_score_drift_p99"] > maximum_score_drift:
        raise RuntimeError(f"{backend} score-drift validation failed: {report}")
    if report["top1_agreement"] < minimum_top1_agreement:
        raise RuntimeError(f"{backend} decision validation failed: {report}")
    if (
        "reference_top1_speaker_accuracy" in report
        and report["candidate_top1_speaker_accuracy"]
        < report["reference_top1_speaker_accuracy"] - maximum_top1_accuracy_drop
    ):
        raise RuntimeError(f"{backend} speaker-accuracy validation failed: {report}")
    if abs(report["minimum_output_norm"] - 1.0) > 1e-5 or abs(report["maximum_output_norm"] - 1.0) > 1e-5:
        raise RuntimeError(f"{backend} output is not normalized: {report}")


def _ort_session(path: Path, device: torch.device) -> Any:
    ort = importlib.import_module("onnxruntime")
    providers: list[Any] = ["CPUExecutionProvider"]
    if device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        providers.insert(
            0,
            ("CUDAExecutionProvider", {"device_id": device.index}) if device.index else "CUDAExecutionProvider",
        )
    return ort.InferenceSession(str(path), providers=providers)


def _validate_session(
    path: Path,
    reference: torch.nn.Module,
    device: torch.device,
    *,
    seed: int,
) -> dict[str, Any]:
    session = _ort_session(path, device)
    rng = np.random.default_rng(seed)
    expected_all: list[np.ndarray] = []
    actual_all: list[np.ndarray] = []
    with torch.inference_mode():
        for frames in VALIDATION_FRAMES:
            for _ in range(3):
                features = rng.standard_normal((1, 1, 72, frames), dtype=np.float32)
                expected = reference(torch.from_numpy(features).to(device)).float().cpu().numpy()
                actual = session.run(["embedding"], {"features": features})[0]
                expected_all.append(expected[0])
                actual_all.append(actual[0])
    report = _numerical_report(np.stack(expected_all), np.stack(actual_all))
    _require_parity(report, 0.99999, 1e-4, "ONNX")
    return report


def _reference_waveforms(  # pylint: disable=too-many-locals
    audio_dir: Path,
    duration_seconds: float,
    max_speakers: int,
    max_recordings_per_speaker: int,
) -> ReferenceAudio:
    if max_speakers < 0 or max_speakers == 1 or max_recordings_per_speaker < 0 or max_recordings_per_speaker == 1:
        raise ValueError("speaker and recording limits must be zero for all or at least two")
    soundfile = importlib.import_module("soundfile")
    sample_count = round(duration_seconds * 16_000)
    waveforms: list[torch.Tensor] = []
    speaker_ids: list[str] = []
    speaker_count = 0
    for speaker_dir in sorted(path for path in audio_dir.iterdir() if path.is_dir() and not path.name.startswith("_")):
        candidates: list[Path] = []
        for path in sorted(speaker_dir.rglob("*")):
            if path.suffix.lower() not in {".flac", ".ogg", ".wav"}:
                continue
            try:
                info = soundfile.info(path)
            except RuntimeError:
                continue
            if info.samplerate == 16_000 and info.frames >= sample_count:
                candidates.append(path)
            if max_recordings_per_speaker and len(candidates) == max_recordings_per_speaker:
                break
        if len(candidates) < 2:
            continue
        for path in candidates:
            values, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
            if sample_rate != 16_000:
                raise RuntimeError(f"reference audio sample rate changed while reading {path}")
            mono = values.mean(axis=1)
            offset = (len(mono) - sample_count) // 2
            crop = np.array(mono[offset : offset + sample_count], dtype=np.float32, copy=True)
            waveforms.append(torch.from_numpy(crop))
            speaker_ids.append(speaker_dir.name)
        speaker_count += 1
        if max_speakers and speaker_count == max_speakers:
            break
    if speaker_count < 2:
        raise RuntimeError(
            f"{audio_dir} needs at least two speaker folders with two 16 kHz recordings of {duration_seconds:g}s"
        )
    return ReferenceAudio(tuple(waveforms), tuple(speaker_ids))


def _upstream_waveform_embeddings(
    model: torch.nn.Module,
    waveforms: Sequence[torch.Tensor],
    device: torch.device,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for waveform in waveforms:
            output = model(waveform.unsqueeze(0).to(device))
            output = torch.nn.functional.normalize(output.float().reshape(1, -1), dim=1)
            embeddings.append(output.cpu().numpy()[0])
    return np.stack(embeddings)


def _onnx_waveform_embeddings(session: Any, waveforms: Sequence[torch.Tensor], device: torch.device) -> np.ndarray:
    frontend_module = importlib.import_module("redimnet2_onnx.frontend")
    frontend = frontend_module.ReDimNetFrontend().to(device).eval()
    embeddings: list[np.ndarray] = []
    with torch.inference_mode():
        for waveform in waveforms:
            features = frontend(waveform.unsqueeze(0).to(device)).cpu().numpy()
            embeddings.append(session.run(["embedding"], {"features": features})[0][0])
    return np.stack(embeddings)


def validate_tensorrt_against_reference(  # pylint: disable=too-many-locals
    path: str | Path,
    spec: ModelSpec,
    audio_dir: str | Path,
    *,
    device_id: int = 0,
    max_speakers: int = 8,
    max_recordings_per_speaker: int = 2,
    progress: Callable[[float, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    tensorrt_module = importlib.import_module("redimnet2_onnx.tensorrt")
    target = torch.device(f"cuda:{device_id}")
    model = _load_upstream_model(spec, target)
    model_path = Path(path)
    onnx_session = _ort_session(model_path, target)
    runtime = tensorrt_module.TensorRTReDimNet2(model_path, device_id=device_id)
    duration_reports: dict[str, Any] = {}
    validation_errors: list[str] = []
    for duration in (5.0, 10.0, 30.0):
        audio = _reference_waveforms(Path(audio_dir), duration, max_speakers, max_recordings_per_speaker)
        expected = _upstream_waveform_embeddings(model, audio.waveforms, target)
        onnx_actual = _onnx_waveform_embeddings(onnx_session, audio.waveforms, target)
        tensorrt_actual = np.stack(
            [runtime.embed(waveform.unsqueeze(0)).cpu().numpy()[0] for waveform in audio.waveforms]
        )
        report = _numerical_report(expected, tensorrt_actual, audio.speaker_ids)
        onnx_report = _numerical_report(expected, onnx_actual, audio.speaker_ids)
        tensorrt_delta_report = _numerical_report(onnx_actual, tensorrt_actual, audio.speaker_ids)
        report["comparisons"] = {
            "onnx_fp32_vs_official_pytorch_fp32": onnx_report,
            "tensorrt_fp16_vs_onnx_fp32": tensorrt_delta_report,
        }
        report.update({"speakers": len(set(audio.speaker_ids)), "recordings": len(audio.waveforms)})
        if progress is not None:
            progress(duration, report)
        try:
            _require_parity(
                onnx_report,
                ONNX_REFERENCE_MINIMUM_COSINE,
                ONNX_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "ONNX FP32 vs official PyTorch FP32",
                minimum_top1_agreement=REFERENCE_MINIMUM_TOP1_AGREEMENT,
                maximum_top1_accuracy_drop=REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            )
            _require_parity(
                tensorrt_delta_report,
                TENSORRT_REFERENCE_MINIMUM_COSINE,
                TENSORRT_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "TensorRT FP16 vs ONNX FP32",
                minimum_top1_agreement=REFERENCE_MINIMUM_TOP1_AGREEMENT,
                maximum_top1_accuracy_drop=REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            )
            _require_parity(
                report,
                TENSORRT_REFERENCE_MINIMUM_COSINE,
                TENSORRT_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "TensorRT FP16 vs official PyTorch FP32",
                minimum_top1_agreement=REFERENCE_MINIMUM_TOP1_AGREEMENT,
                maximum_top1_accuracy_drop=REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            )
        except RuntimeError as error:
            validation_errors.append(f"{duration:g}s: {error}")
        duration_reports[f"{duration:g}"] = report
    if validation_errors:
        raise RuntimeError("; ".join(validation_errors))
    return {"contract": _contract(model_path), "durations": duration_reports}


def _source_checkpoint_sha256(spec: ModelSpec) -> str:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / spec.source_filename
    if not checkpoint.is_file():
        raise RuntimeError(f"upstream checkpoint was not retained at expected path: {checkpoint}")
    return file_sha256(checkpoint)


def export_model(spec: ModelSpec, output_dir: Path, device: torch.device) -> dict[str, Any]:
    model = _load_upstream_model(spec, device)
    wrapper = NormalizedMelEmbedding(model).to(device).eval()
    output_path = output_dir / spec.filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros((1, 1, 72, 996), dtype=torch.float32, device=device)
    started = time.perf_counter()
    torch.onnx.export(
        wrapper,
        (example,),
        output_path,
        input_names=["features"],
        output_names=["embedding"],
        opset_version=18,
        dynamo=True,
        external_data=False,
        dynamic_shapes=_dynamic_shapes(),
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    stable_seed = int.from_bytes(hashlib.sha256(spec.name.encode()).digest()[:4], "little")
    report = {
        "name": spec.name,
        "filename": spec.filename,
        "sha256": file_sha256(output_path),
        "bytes": output_path.stat().st_size,
        "model_name": spec.model_name,
        "dataset": spec.dataset,
        "train_type": spec.train_type,
        "source_url": spec.source_url,
        "source_sha256": _source_checkpoint_sha256(spec),
        "upstream_commit": UPSTREAM_COMMIT,
        "export_seconds": time.perf_counter() - started,
        "contract": _contract(output_path),
        "numerical": _validate_session(output_path, wrapper, device, seed=stable_seed),
    }
    output_path.with_suffix(".export.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _load_existing_report(spec: ModelSpec, output_dir: Path) -> dict[str, Any] | None:
    output_path = output_dir / spec.filename
    report_path = output_path.with_suffix(".export.json")
    if not output_path.is_file() or not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected_metadata = {
        "name": spec.name,
        "filename": spec.filename,
        "model_name": spec.model_name,
        "dataset": spec.dataset,
        "train_type": spec.train_type,
        "source_url": spec.source_url,
        "upstream_commit": UPSTREAM_COMMIT,
    }
    if any(report.get(key) != value for key, value in expected_metadata.items()):
        return None
    if report.get("bytes") != output_path.stat().st_size or report.get("sha256") != file_sha256(output_path):
        return None
    if not isinstance(report.get("contract"), dict) or not isinstance(report.get("numerical"), dict):
        return None
    return report


def _write_manifest(output_dir: Path, reports: list[dict[str, Any]]) -> Path:
    manifest = {
        "schema_version": 1,
        "release": MODEL_RELEASE,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "models": {
            report["name"]: {
                key: report[key]
                for key in (
                    "filename",
                    "sha256",
                    "bytes",
                    "model_name",
                    "dataset",
                    "train_type",
                    "source_url",
                    "source_sha256",
                    "upstream_commit",
                )
            }
            for report in reports
        },
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export pinned ReDimNet2 checkpoints to normalized dynamic ONNX")
    parser.add_argument("output_dir", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="export every supported checkpoint")
    group.add_argument("--model", choices=available_models(), help="export one model")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="reuse previously validated exports")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    names = available_models() if args.all else (args.model,)
    reports = []
    for index, name in enumerate(names, start=1):
        if name is None:
            raise RuntimeError("a model name is required")
        spec = get_model_spec(name)
        if args.resume and (report := _load_existing_report(spec, args.output_dir)) is not None:
            print(f"[{index}/{len(names)}] reusing {name}", flush=True)
            reports.append(report)
            continue
        print(f"[{index}/{len(names)}] exporting {name}", flush=True)
        reports.append(export_model(spec, args.output_dir, device))
    manifest = _write_manifest(args.output_dir, reports)
    print(f"Wrote {len(reports)} model(s) and {manifest}")


if __name__ == "__main__":
    main()
