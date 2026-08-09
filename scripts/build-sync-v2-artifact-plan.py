#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def state_at(structure: dict, timestamp_ms: int) -> str:
    for segment in structure.get("activitySegments", []):
        r = segment["range"]
        if int(r["start"]) <= timestamp_ms < int(r["endExclusive"]):
            return str(segment["kind"])
    return "UNKNOWN"


def representative_rows(normalized: dict, frozen: dict, limit: int = 10) -> list[dict]:
    rows = list(normalized.get("mappings", []))
    selected: list[dict] = []

    # Preserve explicit unmatched timestamp decisions first.
    selected.extend(row for row in rows if row.get("reference_ms") is None)

    matched = [row for row in rows if row.get("reference_ms") is not None]
    by_unit: dict[str, list[dict]] = {}
    for row in matched:
        by_unit.setdefault(str(row.get("source_unit_id", "unknown")), []).append(row)
    for unit_rows in by_unit.values():
        for index in {0, len(unit_rows) // 2, len(unit_rows) - 1}:
            selected.append(unit_rows[index])

    # Refused/source-unmatched units have no timestamp timeline in the frozen contract.
    source_units = {unit["unitId"]: unit for unit in frozen.get("sourceTemporalStructure", {}).get("motionUnits", [])}
    known_unmatched = {
        str(item.get("source_unit_id") or item.get("unit_id"))
        for item in normalized.get("unmatched_source_units", [])
        if item.get("source_unit_id") or item.get("unit_id")
    }
    for unit_id in sorted(known_unmatched):
        unit = source_units.get(unit_id)
        if not unit:
            continue
        r = unit["range"]
        timestamp = (int(r["start"]) + int(r["endExclusive"])) // 2
        selected.append({
            "source_ms": timestamp,
            "reference_ms": None,
            "confidence": float(normalized.get("confidence", 0.0)),
            "source_unit_id": unit_id,
            "reference_unit_id": None,
            "source_direction": "UNKNOWN",
            "reference_direction": "UNMATCHED",
            "source_phase": unit.get("structureClass", "UNKNOWN"),
            "reference_phase": "UNMATCHED",
            "source_state": state_at(frozen.get("sourceTemporalStructure", {}), timestamp),
            "reference_state": "UNMATCHED",
            "reliability": "UNRELIABLE" if normalized.get("status") == "REFUSED" else "UNKNOWN",
        })

    # Deduplicate and deterministically spread over the timeline.
    unique: dict[tuple, dict] = {}
    for row in selected:
        key = (row.get("source_ms"), row.get("reference_ms"), row.get("source_unit_id"))
        unique[key] = row
    ordered = sorted(unique.values(), key=lambda row: (int(row.get("source_ms", 0)), str(row.get("source_unit_id", ""))))
    if len(ordered) <= limit:
        return ordered
    indexes = sorted({round(i * (len(ordered) - 1) / (limit - 1)) for i in range(limit)})
    return [ordered[i] for i in indexes]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--normalized", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    normalized = json.loads(Path(args.normalized).read_text(encoding="utf-8"))
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    spatial = dict(normalized.get("spatial_diagnostics", {}))
    spatial.pop("frozen_diagnostics", None)
    source_scale = spatial.get("source_uniform_scale")
    reference_scale = spatial.get("reference_uniform_scale")
    if isinstance(source_scale, (int, float)) and isinstance(reference_scale, (int, float)) and reference_scale > 0:
        spatial["uniform_scale"] = source_scale / reference_scale
    spatial["transform_family"] = "similarity_only"
    spatial["review_note"] = "Production Sync-v2 Android output. Form differences are not scored or non-rigidly normalized."
    plan = {
        "schema_version": 1,
        "origin": "production_sync_v2_android_output",
        "case_id": args.case,
        "purpose": "Human inspection of actual Sync-v2 mapping/refusal evidence; rows are selected from production output, not a golden path.",
        "production_status": normalized.get("status"),
        "confidence": normalized.get("confidence"),
        "source_analyzable_fraction": normalized.get("source_analyzable_fraction"),
        "reference_analyzable_fraction": normalized.get("reference_analyzable_fraction"),
        "refusal_reason": normalized.get("refusal_reason"),
        "spatial_diagnostics": spatial,
        "rows": representative_rows(normalized, frozen),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
