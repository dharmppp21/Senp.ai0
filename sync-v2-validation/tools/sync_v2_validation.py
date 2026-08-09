#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
from functools import lru_cache
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TOOL_DIR = Path(__file__).resolve().parent
MODULE_ROOT = TOOL_DIR.parent
REPO_ROOT = MODULE_ROOT.parent
QA_ROOT = REPO_ROOT / "qa"
if str(QA_ROOT) not in sys.path:
    sys.path.insert(0, str(QA_ROOT))

from senpqa.visuals import FramePoint, _extract_frame, _font_path, _probe_frames, _stack_images

DEFAULT_CASES = MODULE_ROOT / "fixtures" / "real-video-cases.json"
DEFAULT_REVIEW = MODULE_ROOT / "fixtures" / "legraise-renderer-adversarial-review.json"
BUDGET_FRACTION = 0.20


class ValidationError(RuntimeError):
    pass


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(case_id: str, cases_path: Path = DEFAULT_CASES) -> tuple[dict[str, Any], dict[str, Any]]:
    root = load_json(cases_path)
    cases = [item for item in root["cases"] if item["id"] == case_id]
    if len(cases) != 1:
        raise ValidationError(f"unknown or duplicate real-video case: {case_id}")
    return root, cases[0]


def manifest_index(path: Path) -> dict[str, dict[str, Any]]:
    manifest = load_json(path)
    return {item["relative_path"]: item for item in manifest["videos"]}


def validate_case_files(root: dict[str, Any], case: dict[str, Any], verify_hash: bool = True) -> dict[str, Any]:
    corpus_root = Path(root["corpus_root"])
    manifest_path = Path(root["corpus_manifest"])
    if not corpus_root.is_dir() or not manifest_path.is_file():
        raise ValidationError("external corpus or manifest is unavailable")
    index = manifest_index(manifest_path)
    reports: dict[str, Any] = {}
    for role in ("source", "reference"):
        relative = case[role]
        path = corpus_root / relative
        record = index.get(relative)
        if record is None or not path.is_file():
            raise ValidationError(f"corpus case file is not locked by manifest: {relative}")
        actual_hash = sha256(path) if verify_hash else record["sha256"]
        if actual_hash != record["sha256"]:
            raise ValidationError(f"corpus hash mismatch for {relative}")
        reports[role] = {
            "path": str(path.resolve()),
            "relative_path": relative,
            "sha256": record["sha256"],
            "duration_ms": round(float(record["duration_sec"]) * 1000),
            "codec": record["video_codec"],
            "width": record["width"],
            "height": record["height"],
            "input_nominal_fps": record["avg_frame_rate"],
        }
    return reports


def prepare_real(args: argparse.Namespace) -> dict[str, Any]:
    root, case = load_case(args.case, Path(args.cases))
    files = validate_case_files(root, case, verify_hash=not args.no_hash)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    normalized_output = output / "normalized-result.json"
    descriptor = {
        "schema_version": 1,
        "protocol": "senp-sync-v2-validation-adapter/1",
        "mode": "real_video",
        "case_id": case["id"],
        "focus": case["focus"],
        "prior_evidence": case.get("prior_evidence"),
        "source": files["source"],
        "reference": files["reference"],
        "analysis_fps": args.analysis_fps,
        "timestamp_truth": "decoded_presentation_timestamps",
        "required_output": {
            "path": str(normalized_output),
            "schema": "normalized-result-v1",
            "must_include": [
                "status", "confidence", "source_analyzable_fraction", "reference_analyzable_fraction",
                "mappings", "unmatched_source_units", "unmatched_reference_units", "spatial_diagnostics",
            ],
        },
        "out_of_scope": ["coaching", "problem_counts", "scores"],
    }
    descriptor_path = output / "adapter-request.json"
    write_json(descriptor_path, descriptor)
    report: dict[str, Any] = {
        "case_id": case["id"],
        "descriptor": str(descriptor_path),
        "normalized_result": str(normalized_output),
        "integration_status": "STAGED",
        "files": files,
    }
    if args.adapter_executable:
        executable = Path(args.adapter_executable).resolve()
        if not executable.is_file():
            raise ValidationError(f"adapter executable not found: {executable}")
        started = time.perf_counter_ns()
        completed = subprocess.run([str(executable), str(descriptor_path)], text=True, capture_output=True, check=False)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        report["adapter_elapsed_ms"] = elapsed_ms
        report["adapter_stdout"] = completed.stdout[-4000:]
        report["adapter_stderr"] = completed.stderr[-4000:]
        if completed.returncode != 0:
            raise ValidationError(f"adapter failed with exit code {completed.returncode}")
        if not normalized_output.is_file():
            raise ValidationError("adapter succeeded but did not create normalized-result.json")
        report["validation"] = validate_normalized(case, load_json(normalized_output))
        report["integration_status"] = "EXECUTED"
    write_json(output / "real-run-report.json", report)
    return report


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(str(key).lower())
            keys.extend(_walk_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_walk_keys(child))
    return keys


def _opposite(left: Any, right: Any) -> bool:
    numeric = {(-1, 1), (1, -1), (-1.0, 1.0), (1.0, -1.0)}
    if (left, right) in numeric:
        return True
    pair = (str(left).upper(), str(right).upper())
    return pair in {
        ("POSITIVE", "NEGATIVE"), ("NEGATIVE", "POSITIVE"),
        ("RAISING", "LOWERING"), ("LOWERING", "RAISING"),
        ("ASCENDING", "DESCENDING"), ("DESCENDING", "ASCENDING"),
    }


def validate_normalized(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"score", "scores", "coaching", "problems", "problem_count", "error_count"}
    present_forbidden = sorted(forbidden.intersection(_walk_keys(result)))
    if present_forbidden:
        raise ValidationError(f"scoring/coaching fields are out of Sync-v2 validation scope: {present_forbidden}")
    status = result.get("status")
    if status not in {"SYNCHRONIZED", "PARTIAL", "REFUSED"}:
        raise ValidationError(f"invalid synchronization status: {status}")
    for key in ("confidence", "source_analyzable_fraction", "reference_analyzable_fraction"):
        value = result.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValidationError(f"{key} must be a finite probability")
    mappings = result.get("mappings", [])
    if not isinstance(mappings, list):
        raise ValidationError("mappings must be a list")
    for key in ("unmatched_source_units", "unmatched_reference_units"):
        if not isinstance(result.get(key), list):
            raise ValidationError(f"{key} must be an explicit list, even when empty")
    groups: dict[str, list[dict[str, Any]]] = {}
    opposite_rows = 0
    forced_hole_rows = 0
    for row in mappings:
        source_ms = row.get("source_ms")
        reference_ms = row.get("reference_ms")
        confidence = row.get("confidence")
        if not isinstance(source_ms, int) or source_ms < 0:
            raise ValidationError("every mapping requires a non-negative integer source_ms")
        if reference_ms is not None and (not isinstance(reference_ms, int) or reference_ms < 0):
            raise ValidationError("reference_ms must be null/UNMATCHED or a non-negative integer")
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            raise ValidationError("mapping confidence must be in [0, 1]")
        unit = str(row.get("source_unit_id", "__unassigned__"))
        groups.setdefault(unit, []).append(row)
        reliable = str(row.get("reliability", "RELIABLE")).upper() == "RELIABLE"
        if reliable and reference_ms is not None and _opposite(row.get("source_direction"), row.get("reference_direction")):
            opposite_rows += 1
        states = {str(row.get("source_state", "")).upper(), str(row.get("reference_state", "")).upper()}
        unavailable_state = any(
            token in state
            for state in states
            for token in ("REST", "UNRELIABLE", "COVERAGE_HOLE")
        )
        if reference_ms is not None and confidence >= 0.5 and unavailable_state:
            forced_hole_rows += 1
    for unit, rows in groups.items():
        source_times = [row["source_ms"] for row in rows]
        if any(a >= b for a, b in zip(source_times, source_times[1:])):
            raise ValidationError(f"source timestamps are not strictly increasing inside normalized unit {unit}")
        ref_times = [row["reference_ms"] for row in rows if row["reference_ms"] is not None]
        if any(a > b for a, b in zip(ref_times, ref_times[1:])):
            raise ValidationError(f"reference timestamps decrease inside normalized unit {unit}")
    if opposite_rows:
        raise ValidationError(f"reliable interior/directional mapping contains {opposite_rows} opposite-direction rows")
    if forced_hole_rows:
        raise ValidationError(f"found {forced_hole_rows} confident mappings through rest/unreliable coverage")
    spatial = result.get("spatial_diagnostics")
    if not isinstance(spatial, dict):
        raise ValidationError("spatial_diagnostics must be explicit")
    spatial_keys = set(_walk_keys(spatial))
    forbidden_spatial = {"shear", "non_uniform_scale", "scale_x", "scale_y", "scale_z"}
    invalid_spatial = sorted(spatial_keys.intersection(forbidden_spatial))
    if invalid_spatial:
        raise ValidationError(f"frozen spatial family forbids non-rigid transform fields: {invalid_spatial}")
    if status == "REFUSED" and not result.get("refusal_reason"):
        raise ValidationError("REFUSED output requires refusal_reason")
    review: list[str] = []
    min_coverage = min(float(result["source_analyzable_fraction"]), float(result["reference_analyzable_fraction"]))
    minimum_for_synchronized = case.get("minimum_analyzable_fraction_for_synchronized")
    if minimum_for_synchronized is not None:
        minimum_for_synchronized = float(minimum_for_synchronized)
        if not 0.0 <= minimum_for_synchronized <= 1.0:
            raise ValidationError("case minimum_analyzable_fraction_for_synchronized must be in [0, 1]")
        if status == "SYNCHRONIZED" and min_coverage < minimum_for_synchronized:
            raise ValidationError(
                f"truthfulness target cannot report SYNCHRONIZED at analyzable fraction {min_coverage:.3f} "
                f"below configured minimum {minimum_for_synchronized:.3f}"
            )
    if "low_observation_coverage" in case.get("focus", []) and status == "SYNCHRONIZED":
        review.append(f"low-coverage regression target synchronized at analyzable fraction {min_coverage:.3f}; human review required")
    return {
        "ok": True,
        "case_id": case["id"],
        "status": status,
        "mapping_rows": len(mappings),
        "opposite_direction_rows": opposite_rows,
        "forced_coverage_hole_rows": forced_hole_rows,
        "review_notes": review,
    }


def command_validate_normalized(args: argparse.Namespace) -> dict[str, Any]:
    _, case = load_case(args.case, Path(args.cases))
    report = validate_normalized(case, load_json(Path(args.result)))
    if args.output:
        write_json(Path(args.output), report)
    return report


@lru_cache(maxsize=8)
def _frame_timestamps(video: str) -> tuple[float, ...]:
    return tuple(_probe_frames(Path(video), "ffprobe"))


def _nearest_frame(video: Path, requested_ms: int) -> FramePoint:
    timestamps = _frame_timestamps(str(video.resolve()))
    requested_sec = requested_ms / 1000.0
    index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i] - requested_sec))
    return FramePoint(index, requested_ms, round(timestamps[index] * 1000))


def _placeholder(output: Path, label: str) -> None:
    font = _font_path()
    safe = "".join(ch if ch.isalnum() or ch in " ._/-" else "_" for ch in label)
    filt = (
        "drawtext=fontfile='" + font + "':text='UNMATCHED':x=(w-text_w)/2:y=185:fontsize=34:fontcolor=white," +
        "drawtext=fontfile='" + font + "':text='" + safe + "':x=(w-text_w)/2:y=235:fontsize=13:fontcolor=white"
    )
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=0x141414:s=360x460", "-vf", filt, "-frames:v", "1", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValidationError(f"failed to render UNMATCHED placeholder: {completed.stderr}")


def _video_meta(path: Path, manifest_record: dict[str, Any]) -> tuple[str, str]:
    width = int(manifest_record["width"])
    height = int(manifest_record["height"])
    orientation = "portrait" if height > width else "landscape"
    return str(manifest_record["video_codec"]), orientation


def render_artifact(args: argparse.Namespace) -> dict[str, Any]:
    cases_root, case = load_case(args.case, Path(args.cases))
    files = validate_case_files(cases_root, case, verify_hash=not args.no_hash)
    corpus_root = Path(cases_root["corpus_root"])
    manifest = manifest_index(Path(cases_root["corpus_manifest"]))
    source = corpus_root / case["source"]
    reference = corpus_root / case["reference"]
    source_codec, source_orientation = _video_meta(source, manifest[case["source"]])
    reference_codec, reference_orientation = _video_meta(reference, manifest[case["reference"]])
    plan = load_json(Path(args.plan))
    normalized_validation = validate_normalized(case, plan) if "status" in plan else None
    rows = plan.get("rows", plan.get("mappings", []))
    if not rows:
        raise ValidationError("artifact plan contains no mapping rows")
    output = Path(args.output_dir).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    row_images: list[Path] = []
    rendered_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_point = _nearest_frame(source, int(row["source_ms"]))
        source_frame = output / f"row-{index:02d}-source.png"
        source_label = (
            f"SOURCE {source_point.actual_ms}ms | {row.get('source_unit_id', '?')} | "
            f"{row.get('source_direction', '?')} / {row.get('source_phase', '?')} / {row.get('source_state', '?')} | "
            f"conf {float(row.get('confidence', 0.0)):.2f}"
        )
        _extract_frame(
            video=source,
            output=source_frame,
            point=source_point,
            label=source_label,
            codec=source_codec,
            orientation=source_orientation,
            role="source",
            ffmpeg="ffmpeg",
        )
        reference_frame = output / f"row-{index:02d}-reference.png"
        reference_ms = row.get("reference_ms")
        if reference_ms is None:
            _placeholder(reference_frame, "reference timestamp = UNMATCHED")
            actual_reference_ms = None
        else:
            reference_point = _nearest_frame(reference, int(reference_ms))
            actual_reference_ms = reference_point.actual_ms
            reference_label = (
                f"REFERENCE {reference_point.actual_ms}ms | {row.get('reference_unit_id', '?')} | "
                f"{row.get('reference_direction', '?')} / {row.get('reference_phase', '?')} / {row.get('reference_state', '?')}"
            )
            _extract_frame(
                video=reference,
                output=reference_frame,
                point=reference_point,
                label=reference_label,
                codec=reference_codec,
                orientation=reference_orientation,
                role="reference",
                ffmpeg="ffmpeg",
            )
        row_image = output / f"row-{index:02d}-pair.png"
        _stack_images([source_frame, reference_frame], row_image, columns=2, ffmpeg="ffmpeg")
        row_images.append(row_image)
        rendered_rows.append({
            **row,
            "source_actual_ms": source_point.actual_ms,
            "reference_actual_ms": actual_reference_ms,
            "source_frame": source_frame.name,
            "reference_frame": reference_frame.name,
            "pair_image": row_image.name,
        })
    sheet = output / "contact-sheet.png"
    _stack_images(row_images, sheet, columns=1, ffmpeg="ffmpeg")
    timeline = output / "timeline.svg"
    _render_timeline_svg(
        rendered_rows,
        source_duration_ms=int(files["source"]["duration_ms"]),
        reference_duration_ms=int(files["reference"]["duration_ms"]),
        output=timeline,
    )
    page = output / "index.html"
    page.write_text(_render_html(case, plan, rendered_rows, timeline.name), encoding="utf-8")
    report = {
        "schema_version": 1,
        "case_id": case["id"],
        "origin": plan.get("origin", "normalized_adapter_output"),
        "purpose": plan.get("purpose"),
        "source": files["source"],
        "reference": files["reference"],
        "rows": rendered_rows,
        "spatial_diagnostics": plan.get("spatial_diagnostics", {}),
        "normalized_validation": normalized_validation,
        "contact_sheet": str(sheet),
        "timeline": str(timeline),
        "html": str(page),
    }
    write_json(output / "artifact-report.json", report)
    return report


def _render_timeline_svg(
    rows: list[dict[str, Any]],
    source_duration_ms: int,
    reference_duration_ms: int,
    output: Path,
) -> None:
    width = 960
    left = 92
    right = 36
    span = width - left - right
    source_y = 66
    reference_y = 154

    def position(timestamp_ms: int, duration_ms: int) -> float:
        duration = max(duration_ms, 1)
        return left + span * min(max(timestamp_ms / duration, 0.0), 1.0)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="205" viewBox="0 0 {width} 205">',
        '<rect width="100%" height="100%" fill="#0d0f10"/>',
        '<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" fill="#f4f1e8">',
        '<text x="18" y="20" font-size="13" font-weight="700">TIMESTAMP CORRESPONDENCE</text>',
        f'<text x="18" y="70" font-size="11">SOURCE</text><text x="18" y="158" font-size="11">REF</text>',
        f'<line x1="{left}" y1="{source_y}" x2="{left + span}" y2="{source_y}" stroke="#7d8080" stroke-width="2"/>',
        f'<line x1="{left}" y1="{reference_y}" x2="{left + span}" y2="{reference_y}" stroke="#7d8080" stroke-width="2"/>',
        f'<text x="{left}" y="190" font-size="10">0 ms</text>',
        f'<text x="{left + span}" y="190" text-anchor="end" font-size="10">src {source_duration_ms} ms / ref {reference_duration_ms} ms</text>',
    ]
    for index, row in enumerate(rows, 1):
        source_ms = int(row["source_actual_ms"])
        source_x = position(source_ms, source_duration_ms)
        reference_ms = row.get("reference_actual_ms")
        elements.append(f'<circle cx="{source_x:.2f}" cy="{source_y}" r="7" fill="#0d0f10" stroke="#f4f1e8" stroke-width="2"/>')
        elements.append(f'<text x="{source_x:.2f}" y="{source_y - 12}" text-anchor="middle" font-size="10">{index}</text>')
        if reference_ms is None:
            elements.append(f'<line x1="{source_x:.2f}" y1="{source_y + 9}" x2="{source_x:.2f}" y2="{reference_y - 15}" stroke="#f4f1e8" stroke-dasharray="5 5"/>')
            elements.append(f'<text x="{source_x:.2f}" y="{reference_y - 2}" text-anchor="middle" font-size="9">UNMATCHED</text>')
        else:
            reference_x = position(int(reference_ms), reference_duration_ms)
            elements.append(f'<line x1="{source_x:.2f}" y1="{source_y + 8}" x2="{reference_x:.2f}" y2="{reference_y - 8}" stroke="#aeb0ad" stroke-width="1.5"/>')
            elements.append(f'<circle cx="{reference_x:.2f}" cy="{reference_y}" r="7" fill="#0d0f10" stroke="#f4f1e8" stroke-width="2"/>')
            elements.append(f'<text x="{reference_x:.2f}" y="{reference_y + 20}" text-anchor="middle" font-size="10">{index}</text>')
    elements.append('</g></svg>')
    output.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _render_html(case: dict[str, Any], plan: dict[str, Any], rows: list[dict[str, Any]], timeline_file: str) -> str:
    spatial = plan.get("spatial_diagnostics", {})
    spatial_json = html.escape(json.dumps(spatial, indent=2, sort_keys=True))
    spatial_summary = " · ".join(
        html.escape(f"{label} {spatial.get(key, '—')}")
        for key, label in (
            ("mirror", "MIRROR"),
            ("selected_side", "SIDE"),
            ("relative_yaw_degrees", "YAW"),
            ("relative_elevation_degrees", "ELEV"),
            ("uniform_scale", "GLOBAL SCALE"),
            ("transform_family", "TRANSFORM"),
        )
    )
    origin = html.escape(str(plan.get("origin", "adapter_output")))
    purpose = html.escape(str(plan.get("purpose", "Human verification of mapped timestamps")))
    production_status = html.escape(str(plan.get("production_status", "UNKNOWN")))
    production_confidence = plan.get("confidence")
    source_coverage = plan.get("source_analyzable_fraction")
    reference_coverage = plan.get("reference_analyzable_fraction")
    refusal_reason = str(plan.get("refusal_reason") or "—")
    production_summary = html.escape(
        f"STATUS {production_status} · CONF {float(production_confidence):.3f} · "
        f"COVERAGE src {float(source_coverage):.3f} / ref {float(reference_coverage):.3f} · REFUSAL {refusal_reason}"
    ) if all(isinstance(value, (int, float)) for value in (production_confidence, source_coverage, reference_coverage)) else f"STATUS {production_status} · REFUSAL {refusal_reason}"
    cards = []
    for index, row in enumerate(rows):
        ref = "UNMATCHED" if row.get("reference_ms") is None else f"{row.get('reference_actual_ms')} ms"
        expectation = html.escape(str(row.get("review_expected", "VERIFY")))
        metadata = html.escape(
            f"src {row.get('source_actual_ms')} ms → ref {ref} | conf {float(row.get('confidence', 0.0)):.2f} | "
            f"units {row.get('source_unit_id')} → {row.get('reference_unit_id')} | "
            f"dir {row.get('source_direction')} → {row.get('reference_direction')} | "
            f"phase {row.get('source_phase')} → {row.get('reference_phase')} | "
            f"state {row.get('source_state')} → {row.get('reference_state')}"
        )
        cards.append(f"""
        <article class="mapping-card">
          <header><span class="row-id">PAIR {index + 1:02d}</span><span class="expectation">{expectation}</span></header>
          <div class="frames">
            <figure><img src="{html.escape(row['source_frame'])}" alt="Source mapped frame"><figcaption>SOURCE</figcaption></figure>
            <figure><img src="{html.escape(row['reference_frame'])}" alt="Reference mapped frame"><figcaption>REFERENCE</figcaption></figure>
          </div>
          <p class="metadata">{metadata}</p>
        </article>""")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sync-v2 human verification — {html.escape(case['id'])}</title>
<style>
:root {{ color-scheme: dark; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background:#0d0f10; color:#f4f1e8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:#0d0f10; }} main {{ width:min(100%, 980px); margin:0 auto; padding:18px 14px 56px; }}
.eyebrow {{ letter-spacing:.12em; text-transform:uppercase; font-size:11px; color:#b7b8b3; }} h1 {{ font:700 clamp(25px,8vw,46px)/1.02 system-ui,sans-serif; margin:8px 0 10px; max-width:14ch; }}
.lede {{ color:#c9c9c2; line-height:1.55; max-width:72ch; }} .origin {{ display:inline-block; border:1px solid #6c6f70; padding:6px 8px; margin:8px 0 12px; font-size:11px; }}
.production-summary {{ border:1px solid #6c6f70; padding:8px 9px; margin:0 0 10px; font-size:11px; line-height:1.5; overflow-wrap:anywhere; }}
.spatial-summary {{ border-block:1px solid #474a4b; padding:9px 0; margin:0 0 20px; color:#d7d6cf; font-size:10px; line-height:1.55; overflow-wrap:anywhere; }}
.mapping-card {{ border-top:1px solid #474a4b; padding:16px 0 22px; }} .mapping-card header {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; }}
.row-id {{ font-weight:800; }} .expectation {{ font-size:11px; border:1px solid #f4f1e8; padding:4px 6px; text-align:right; }}
.frames {{ display:grid; grid-template-columns:1fr 1fr; gap:8px; }} figure {{ margin:0; min-width:0; }} img {{ display:block; width:100%; height:auto; border:1px solid #343738; }} figcaption {{ padding-top:5px; font-size:10px; color:#a9aaa5; }}
.metadata {{ font-size:11px; line-height:1.55; overflow-wrap:anywhere; color:#d7d6cf; }} .timeline {{ display:block; width:100%; height:auto; border:1px solid #343738; margin:8px 0 22px; }} .diagnostics {{ border:1px solid #474a4b; padding:12px; margin-top:18px; }} pre {{ margin:0; white-space:pre-wrap; overflow-wrap:anywhere; font-size:11px; line-height:1.5; }}
.checks {{ font:600 14px/1.5 system-ui,sans-serif; padding-left:18px; }}
@media (min-width:700px) {{ main {{ padding:34px 28px 72px; }} .mapping-card {{ padding:22px 0 30px; }} .metadata {{ font-size:12px; }} }}
</style></head><body><main>
<p class="eyebrow">SENP.AI0 / SYNCHRONIZATION KERNEL V2 / HUMAN VERIFICATION</p>
<h1>{html.escape(case['id'])}</h1><p class="lede">{purpose}</p><span class="origin">ORIGIN: {origin}</span>
<p class="production-summary">{production_summary}</p>
<p class="spatial-summary">{spatial_summary}</p>
<section><h2>Review checks</h2><ul class="checks"><li>Raised vs flat legs must be visually obvious.</li><li>Opposite motion direction must not hide behind a similar pose.</li><li>UNMATCHED stays visibly blank instead of forcing correspondence through rest or coverage holes.</li><li>Spatial diagnostics expose mirror, side, view and global scale; non-uniform scale or shear is invalid.</li></ul></section>
<section><h2>Timeline</h2><img class="timeline" src="{html.escape(timeline_file)}" alt="Source to reference timestamp correspondence timeline"></section>
{''.join(cards)}
<section class="diagnostics"><h2>Spatial diagnostics</h2><pre>{spatial_json}</pre></section>
</main></body></html>"""


def _legacy_stage_summary(path: Path) -> dict[str, Any]:
    report = load_json(path)
    stages = {item["name"]: float(item["duration_ms"]) for item in report.get("stages", [])}
    pose_ms = sum(value for key, value in stages.items() if key.startswith("video_pose_"))
    post_pose_names = ["motion_source", "phase_source", "motion_reference", "phase_reference", "alignment"]
    post_pose_ms = sum(stages.get(key, 0.0) for key in post_pose_names)
    total_ms = float(report.get("total_duration_ms", sum(stages.values())))
    return {
        "path": str(path.resolve()),
        "run_id": report.get("run_id"),
        "pipeline": "legacy_wave5_evidence_only_not_sync_v2",
        "pose_preprocessing_ms": pose_ms,
        "post_pose_motion_phase_alignment_ms": post_pose_ms,
        "alignment_only_ms": stages.get("alignment"),
        "total_pipeline_ms": total_ms,
        "alignment_fraction_of_total": (stages.get("alignment", 0.0) / total_ms) if total_ms else None,
        "post_pose_fraction_of_total": (post_pose_ms / total_ms) if total_ms else None,
        "peak_rss_bytes": report.get("process_peak_rss_bytes"),
    }


def performance(args: argparse.Namespace) -> dict[str, Any]:
    if args.enforce_budget and not args.adapter_executable:
        raise ValidationError("--enforce-budget requires a concrete --adapter-executable")
    plans = [
        {"id": "sequence-150-coarse", "samples": 150, "analysis_fps": 10, "scenario": "same_video_self"},
        {"id": "sequence-300-typical", "samples": 300, "analysis_fps": 15, "scenario": "same_video_self"},
        {"id": "sequence-600-dense", "samples": 600, "analysis_fps": 30, "scenario": "different_fps"},
        {"id": "sequence-1200-scaling", "samples": 1200, "analysis_fps": 30, "scenario": "variable_speed"},
        {"id": "reference-reuse-1-to-10", "samples": None, "analysis_fps": 15, "scenario": "one_reference_ten_source"},
        {"id": "repeated-units-2-to-7", "samples": None, "analysis_fps": 15, "scenario": "two_reference_seven_source"},
        {"id": "long-idle-rest", "samples": 1200, "analysis_fps": 15, "scenario": "multiple_sets_rests"},
    ]
    measurements = [_legacy_stage_summary(Path(path)) for path in args.stage_report]
    output = Path(args.output).resolve()
    result: dict[str, Any] = {
        "schema_version": 1,
        "target": {
            "metric": "post_pose_sync_fraction_of_total_pipeline",
            "ordinary_clip_budget_fraction": BUDGET_FRACTION,
            "interpretation": "15-20% target band; report-only unless enforce_budget is explicitly enabled with a concrete integrated implementation",
        },
        "input_fps_is_not_analysis_fps": True,
        "benchmark_plan": plans,
        "legacy_baseline_measurements": measurements,
        "sync_v2_measurements": [],
        "sync_v2_integration_status": "STAGED",
        "budget_evaluation": "NOT_APPLICABLE",
        "enforce_budget": bool(args.enforce_budget),
    }
    if args.adapter_executable:
        executable = Path(args.adapter_executable).resolve()
        if not executable.is_file():
            raise ValidationError(f"adapter executable not found: {executable}")
        run_root = output.parent / (output.stem + "-runs")
        run_root.mkdir(parents=True, exist_ok=True)
        v2_measurements: list[dict[str, Any]] = []
        for plan in plans:
            samples: list[float] = []
            total_samples: list[float] = []
            rss: list[int] = []
            raw_measurements: list[dict[str, Any]] = []
            for repetition in range(args.repetitions):
                measurement_path = run_root / f"{plan['id']}-{repetition}.json"
                descriptor_path = run_root / f"{plan['id']}-{repetition}-request.json"
                descriptor = {
                    "schema_version": 1,
                    "protocol": "senp-sync-v2-validation-adapter/1",
                    "mode": "post_pose_benchmark",
                    "case": plan,
                    "repetition": repetition,
                    "result_output": str(measurement_path),
                    "required_result_fields": ["post_pose_sync_ms", "peak_rss_bytes", "total_pipeline_ms"],
                }
                write_json(descriptor_path, descriptor)
                completed = subprocess.run([str(executable), str(descriptor_path)], capture_output=True, text=True, check=False)
                if completed.returncode != 0 or not measurement_path.is_file():
                    raise ValidationError(f"benchmark adapter failed for {plan['id']} repetition {repetition}")
                measured = load_json(measurement_path)
                raw_measurements.append(measured)
                samples.append(float(measured["post_pose_sync_ms"]))
                if measured.get("total_pipeline_ms") is not None:
                    total_samples.append(float(measured["total_pipeline_ms"]))
                rss.append(int(measured["peak_rss_bytes"]))
            median_ms = statistics.median(samples)
            total_pipeline_ms = statistics.median(total_samples) if total_samples else None
            fraction = median_ms / total_pipeline_ms if total_pipeline_ms else None
            v2_measurements.append({
                **plan,
                "repetitions": args.repetitions,
                "post_pose_sync_median_ms": median_ms,
                "post_pose_sync_samples_ms": samples,
                "total_pipeline_samples_ms": total_samples,
                "peak_rss_bytes": max(rss),
                "total_pipeline_ms": total_pipeline_ms,
                "post_pose_sync_fraction_of_total": fraction,
                "source_frames": max(int(item.get("source_frames", 0)) for item in raw_measurements),
                "reference_frames": max(int(item.get("reference_frames", 0)) for item in raw_measurements),
                "coarse_unit_comparisons": max(int(item.get("coarse_unit_comparisons", 0)) for item in raw_measurements),
                "fine_cells_evaluated": max(int(item.get("fine_cells_evaluated", 0)) for item in raw_measurements),
                "maximum_fine_band_width": max(int(item.get("maximum_fine_band_width", 0)) for item in raw_measurements),
                "fine_alignment_count": max(int(item.get("fine_alignment_count", 0)) for item in raw_measurements),
                "naive_whole_video_cells": max(int(item.get("naive_whole_video_cells", 0)) for item in raw_measurements),
                "fine_to_naive_fraction": max(float(item.get("fine_to_naive_fraction", 0.0)) for item in raw_measurements),
            })
        result["sync_v2_measurements"] = v2_measurements
        result["sync_v2_integration_status"] = "EXECUTED"
        evaluated = [item for item in v2_measurements if item["post_pose_sync_fraction_of_total"] is not None]
        over_budget = [item for item in evaluated if item["post_pose_sync_fraction_of_total"] > BUDGET_FRACTION]
        result["budget_evaluation"] = (
            "NOT_APPLICABLE_POST_POSE_ONLY" if not evaluated else "OVER_BUDGET" if over_budget else "WITHIN_BUDGET"
        )
        if args.enforce_budget and not evaluated:
            write_json(output, result)
            raise ValidationError("cannot enforce total-pipeline budget from post-pose-only benchmark measurements")
        if args.enforce_budget and over_budget:
            write_json(output, result)
            raise ValidationError("explicit Sync-v2 performance budget enforcement failed")
    write_json(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronization Kernel v2 validation/media harness")
    sub = parser.add_subparsers(dest="command", required=True)

    real = sub.add_parser("prepare-real")
    real.add_argument("--case", required=True)
    real.add_argument("--cases", default=str(DEFAULT_CASES))
    real.add_argument("--output-dir", required=True)
    real.add_argument("--analysis-fps", type=float, default=15.0)
    real.add_argument("--adapter-executable")
    real.add_argument("--no-hash", action="store_true")
    real.set_defaults(func=prepare_real)

    normalized = sub.add_parser("validate-normalized")
    normalized.add_argument("--case", required=True)
    normalized.add_argument("--cases", default=str(DEFAULT_CASES))
    normalized.add_argument("--result", required=True)
    normalized.add_argument("--output")
    normalized.set_defaults(func=command_validate_normalized)

    artifact = sub.add_parser("artifact")
    artifact.add_argument("--case", required=True)
    artifact.add_argument("--cases", default=str(DEFAULT_CASES))
    artifact.add_argument("--plan", default=str(DEFAULT_REVIEW))
    artifact.add_argument("--output-dir", required=True)
    artifact.add_argument("--no-hash", action="store_true")
    artifact.set_defaults(func=render_artifact)

    perf = sub.add_parser("performance")
    perf.add_argument("--output", required=True)
    perf.add_argument("--stage-report", action="append", default=[])
    perf.add_argument("--adapter-executable")
    perf.add_argument("--repetitions", type=int, default=5)
    perf.add_argument("--enforce-budget", action="store_true")
    perf.set_defaults(func=performance)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = args.func(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValidationError, KeyError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
