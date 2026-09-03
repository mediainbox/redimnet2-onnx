import argparse
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import export_models
import onnxruntime as ort  # pylint: disable=import-error
import torch  # pylint: disable=import-error

from redimnet2_onnx.models import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    ModelSpec,
    available_models,
    file_sha256,
    get_model_spec,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported ONNX models against upstream PyTorch with TensorRT")
    parser.add_argument("artifact_dir", type=Path, help="directory containing the exported ONNX models")
    parser.add_argument("audio_dir", type=Path, help="directory containing one subdirectory per speaker")
    parser.add_argument("--max-speakers", type=int, default=100, help="zero uses every qualifying speaker")
    parser.add_argument(
        "--max-recordings-per-speaker", type=int, default=3, help="zero uses every qualifying recording"
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--model", action="append", choices=available_models(), help="validate only this model; repeatable"
    )
    parser.add_argument("--report", type=Path, help="JSON report path; defaults inside the artifact directory")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse complete measurements from a matching report and evaluate them against the current limits",
    )
    return parser.parse_args()


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _print_comparison(label: str, report: dict[str, Any]) -> None:
    print(
        f"    {label}: cosine={report['minimum_true_cosine']:.8f}, "
        f"P99={report['pairwise_score_drift_p99']:.8f} "
        f"(genuine={report['genuine_pairwise_score_drift_p99']:.8f}, "
        f"impostor={report['impostor_pairwise_score_drift_p99']:.8f}), "
        f"speaker agreement={report['top1_agreement']:.4%}, "
        f"accuracy={report['reference_top1_speaker_accuracy']:.4%}→"
        f"{report['candidate_top1_speaker_accuracy']:.4%}, "
        f"norm={report['minimum_output_norm']:.8f}..{report['maximum_output_norm']:.8f}",
        flush=True,
    )


def _duration_progress(duration: float, report: dict[str, Any]) -> None:
    print(f"  {duration:g}s: {report['speakers']} speakers, {report['recordings']} recordings", flush=True)
    _print_comparison("official FP32 → ONNX FP32", report["comparisons"]["onnx_fp32_vs_official_pytorch_fp32"])
    _print_comparison("ONNX FP32 → TensorRT FP16", report["comparisons"]["tensorrt_fp16_vs_onnx_fp32"])
    _print_comparison("official FP32 → TensorRT FP16", report)
    print(
        f"    diagnostic threshold flips={report['threshold_flip_counts']} ({report['threshold_flip_rate']:.4%}), "
        f"max margin={report['maximum_threshold_flip_reference_margin']:.8f}",
        flush=True,
    )


def _progress_recorder(reports: dict[str, Any]) -> Callable[[float, dict[str, Any]], None]:
    def record(duration: float, report: dict[str, Any]) -> None:
        reports[f"{duration:g}"] = report
        _duration_progress(duration, report)

    return record


def _existing_results(path: Path, configuration: dict[str, Any], resume: bool) -> dict[str, Any]:
    if not resume or not path.is_file():
        return {}
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    previous_configuration = previous.get("configuration")
    if not isinstance(previous_configuration, dict) or not isinstance(previous.get("results"), dict):
        return {}
    previous_measurement = {key: value for key, value in previous_configuration.items() if key != "limits"}
    current_measurement = {key: value for key, value in configuration.items() if key != "limits"}
    if previous_measurement != current_measurement:
        return {}
    return previous["results"]


def _completed_duration_reports(result: dict[str, Any]) -> dict[str, Any] | None:
    validation = result.get("validation")
    reports = validation.get("durations") if isinstance(validation, dict) else result.get("durations")
    if not isinstance(reports, dict) or set(reports) != {"5", "10", "30"}:
        return None
    if not all(isinstance(report, dict) for report in reports.values()):
        return None
    return reports


def _measurement_failures(reports: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    for duration, report in reports.items():
        comparisons = report.get("comparisons")
        if not isinstance(comparisons, dict):
            return [f"{duration}s: comparison metrics are missing"]
        try:
            export_models._require_parity(  # pylint: disable=protected-access
                comparisons["onnx_fp32_vs_official_pytorch_fp32"],
                export_models.ONNX_REFERENCE_MINIMUM_COSINE,
                export_models.ONNX_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "ONNX FP32 vs official PyTorch FP32",
                minimum_top1_agreement=export_models.REFERENCE_MINIMUM_TOP1_AGREEMENT,
                maximum_top1_accuracy_drop=export_models.REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            )
            export_models._require_parity(  # pylint: disable=protected-access
                comparisons["tensorrt_fp16_vs_onnx_fp32"],
                export_models.TENSORRT_REFERENCE_MINIMUM_COSINE,
                export_models.TENSORRT_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "TensorRT FP16 vs ONNX FP32",
                minimum_top1_agreement=export_models.REFERENCE_MINIMUM_TOP1_AGREEMENT,
                maximum_top1_accuracy_drop=export_models.REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            )
            export_models._require_parity(  # pylint: disable=protected-access
                report,
                export_models.TENSORRT_REFERENCE_MINIMUM_COSINE,
                export_models.TENSORRT_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "TensorRT FP16 vs official PyTorch FP32",
                minimum_top1_agreement=export_models.REFERENCE_MINIMUM_TOP1_AGREEMENT,
                maximum_top1_accuracy_drop=export_models.REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            )
        except (KeyError, RuntimeError) as error:
            failures.append(f"{duration}s: {error}")
    return failures


def _review_completed_result(result: dict[str, Any], model_path: Path) -> dict[str, Any] | None:
    reports = _completed_duration_reports(result)
    if reports is None:
        return None
    failures = _measurement_failures(reports)
    if failures:
        return {
            "status": "failed",
            "seconds": result.get("seconds"),
            "error": "; ".join(failures),
            "durations": reports,
            "measurements_reused": True,
        }
    return {
        "status": "passed",
        "seconds": result.get("seconds"),
        "validation": {
            "contract": export_models._contract(model_path),  # pylint: disable=protected-access
            "durations": reports,
        },
        "measurements_reused": True,
    }


def _checkpoint_metadata(artifact_dir: Path, specs: dict[str, ModelSpec]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for name, spec in specs.items():
        report_path = (artifact_dir / spec.filename).with_suffix(".export.json")
        source_sha256 = None
        try:
            export_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
        else:
            if isinstance(export_report.get("source_sha256"), str):
                source_sha256 = export_report["source_sha256"]
        metadata[name] = {"url": spec.source_url, "sha256": source_sha256}
    return metadata


def main() -> None:  # pylint: disable=too-many-locals
    args = _parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    audio_dir = args.audio_dir.expanduser().resolve()
    report_path = (args.report or artifact_dir / "tensorrt-validation.json").expanduser().resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(artifact_dir)
    if not audio_dir.is_dir():
        raise FileNotFoundError(audio_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT validation requires CUDA")

    names = tuple(args.model or available_models())
    specs = {name: get_model_spec(name) for name in names}
    artifacts = {
        name: file_sha256(artifact_dir / spec.filename)
        for name, spec in specs.items()
        if (artifact_dir / spec.filename).is_file()
    }
    if len(artifacts) != len(specs):
        missing = [spec.filename for name, spec in specs.items() if name not in artifacts]
        raise FileNotFoundError(f"missing ONNX artifact(s): {', '.join(missing)}")
    configuration = {
        "artifact_dir": str(artifact_dir),
        "artifact_sha256": artifacts,
        "audio_dir": str(audio_dir),
        "cuda_version": torch.version.cuda,
        "device_id": args.device_id,
        "durations_seconds": [5, 10, 30],
        "gpu": torch.cuda.get_device_name(args.device_id),
        "limits": {
            "onnx_fp32_vs_official_pytorch_fp32": {
                "maximum_pairwise_score_drift_p99": export_models.ONNX_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "minimum_true_cosine": export_models.ONNX_REFERENCE_MINIMUM_COSINE,
                "minimum_top1_speaker_agreement": export_models.REFERENCE_MINIMUM_TOP1_AGREEMENT,
                "maximum_top1_speaker_accuracy_drop": export_models.REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            },
            "tensorrt_fp16": {
                "maximum_pairwise_score_drift_p99": export_models.TENSORRT_REFERENCE_MAXIMUM_SCORE_DRIFT_P99,
                "minimum_true_cosine": export_models.TENSORRT_REFERENCE_MINIMUM_COSINE,
                "minimum_top1_speaker_agreement": export_models.REFERENCE_MINIMUM_TOP1_AGREEMENT,
                "maximum_top1_speaker_accuracy_drop": export_models.REFERENCE_MAXIMUM_TOP1_ACCURACY_DROP,
            },
        },
        "max_recordings_per_speaker": args.max_recordings_per_speaker,
        "max_speakers": args.max_speakers,
        "models": list(names),
        "onnxruntime_version": ort.__version__,
        "tensorrt_version": os.environ.get("TENSORRT_VERSION") or os.environ.get("NVIDIA_TENSORRT_VERSION"),
        "torch_version": torch.__version__,
        "upstream_checkpoint": _checkpoint_metadata(artifact_dir, specs),
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "validator_revision": 4,
    }
    results = _existing_results(report_path, configuration, args.resume)
    release_report: dict[str, Any] = {
        "schema_version": 4,
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration": configuration,
        "results": results,
    }

    for index, name in enumerate(names, start=1):
        if args.resume and isinstance(previous_result := results.get(name), dict):
            reviewed = _review_completed_result(previous_result, artifact_dir / specs[name].filename)
            if reviewed is not None:
                results[name] = reviewed
                print(
                    f"[{index}/{len(names)}] reused complete measurements for {name}: {reviewed['status']}",
                    flush=True,
                )
                release_report["generated_at"] = datetime.now(UTC).isoformat()
                _write_report(report_path, release_report)
                continue
        print(f"[{index}/{len(names)}] validating {name}", flush=True)
        started = time.perf_counter()
        duration_reports: dict[str, Any] = {}
        spec = specs[name]
        try:
            validation = export_models.validate_tensorrt_against_reference(
                artifact_dir / spec.filename,
                spec,
                audio_dir,
                device_id=args.device_id,
                max_speakers=args.max_speakers,
                max_recordings_per_speaker=args.max_recordings_per_speaker,
                progress=_progress_recorder(duration_reports),
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            results[name] = {
                "status": "failed",
                "seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}",
                "durations": duration_reports,
            }
            print(f"  FAILED: {error}", flush=True)
        else:
            results[name] = {
                "status": "passed",
                "seconds": time.perf_counter() - started,
                "validation": validation,
            }
            print(f"  PASSED in {results[name]['seconds']:.1f}s", flush=True)
        release_report["generated_at"] = datetime.now(UTC).isoformat()
        _write_report(report_path, release_report)

    failures = [name for name in names if results.get(name, {}).get("status") != "passed"]
    print(f"Wrote {report_path}", flush=True)
    if failures:
        raise SystemExit(f"TensorRT validation failed for {len(failures)} model(s): {', '.join(failures)}")
    print(f"TensorRT validation passed for all {len(names)} model(s)", flush=True)


if __name__ == "__main__":
    main()
