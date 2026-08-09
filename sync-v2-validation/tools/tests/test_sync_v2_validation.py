from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

TOOL = Path(__file__).resolve().parents[1] / "sync_v2_validation.py"
spec = importlib.util.spec_from_file_location("sync_v2_validation", TOOL)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


class SyncV2MediaHarnessTest(unittest.TestCase):
    def test_locked_real_cases_resolve(self) -> None:
        root = module.load_json(module.DEFAULT_CASES)
        ids = {item["id"] for item in root["cases"]}
        self.assertEqual(
            ids,
            {
                "biceps-wrong-right",
                "legraise-wrong-right",
                "pushup-wrong-right",
                "biceps-right-right-control",
                "legraise-right-right-control",
            },
        )
        manifest = module.manifest_index(Path(root["corpus_manifest"]))
        for case in root["cases"]:
            self.assertIn(case["source"], manifest)
            self.assertIn(case["reference"], manifest)
            self.assertTrue((Path(root["corpus_root"]) / case["source"]).is_file())
            self.assertTrue((Path(root["corpus_root"]) / case["reference"]).is_file())

    def test_normalized_validator_accepts_mapping_without_scoring(self) -> None:
        _, case = module.load_case("biceps-wrong-right")
        result = self.base_result()
        report = module.validate_normalized(case, result)
        self.assertTrue(report["ok"])
        self.assertEqual(report["opposite_direction_rows"], 0)

    def test_normalized_validator_rejects_opposite_direction(self) -> None:
        _, case = module.load_case("biceps-wrong-right")
        result = self.base_result()
        result["mappings"][0]["reference_direction"] = "LOWERING"
        with self.assertRaises(module.ValidationError):
            module.validate_normalized(case, result)

    def test_normalized_validator_rejects_scoring_assumptions(self) -> None:
        _, case = module.load_case("biceps-wrong-right")
        result = self.base_result()
        result["problem_count"] = 2
        with self.assertRaises(module.ValidationError):
            module.validate_normalized(case, result)

    def test_low_coverage_truthfulness_target_rejects_forced_synchronized_success(self) -> None:
        _, case = module.load_case("pushup-wrong-right")
        result = self.base_result()
        result["source_analyzable_fraction"] = 0.31
        result["reference_analyzable_fraction"] = 0.92
        with self.assertRaises(module.ValidationError):
            module.validate_normalized(case, result)

    def test_low_coverage_truthfulness_target_allows_partial(self) -> None:
        _, case = module.load_case("pushup-wrong-right")
        result = self.base_result()
        result["status"] = "PARTIAL"
        result["source_analyzable_fraction"] = 0.31
        result["reference_analyzable_fraction"] = 0.92
        report = module.validate_normalized(case, result)
        self.assertTrue(report["ok"])

    def test_normalized_validator_rejects_forced_coverage_hole_mapping(self) -> None:
        _, case = module.load_case("pushup-wrong-right")
        result = self.base_result()
        result["mappings"][0]["source_state"] = "COVERAGE_HOLE_BOUNDARY"
        with self.assertRaises(module.ValidationError):
            module.validate_normalized(case, result)

    def test_normalized_validator_requires_explicit_unmatched_lists(self) -> None:
        _, case = module.load_case("biceps-wrong-right")
        result = self.base_result()
        result.pop("unmatched_source_units")
        with self.assertRaises(module.ValidationError):
            module.validate_normalized(case, result)

    def test_normalized_validator_rejects_non_rigid_spatial_fields(self) -> None:
        _, case = module.load_case("biceps-wrong-right")
        result = self.base_result()
        result["spatial_diagnostics"]["shear"] = 0.2
        with self.assertRaises(module.ValidationError):
            module.validate_normalized(case, result)

    def test_performance_plan_is_report_only_without_integrated_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "performance.json"
            args = SimpleNamespace(
                enforce_budget=False,
                adapter_executable=None,
                stage_report=[],
                output=str(output),
                repetitions=2,
            )
            report = module.performance(args)
        self.assertEqual(report["sync_v2_integration_status"], "STAGED")
        self.assertEqual(report["budget_evaluation"], "NOT_APPLICABLE")
        self.assertTrue(report["input_fps_is_not_analysis_fps"])
        self.assertEqual(len(report["benchmark_plan"]), 7)
        self.assertEqual({item["analysis_fps"] for item in report["benchmark_plan"]}, {10, 15, 30})
        self.assertTrue(any(item["scenario"] == "one_reference_ten_source" for item in report["benchmark_plan"]))
        self.assertTrue(any(item["scenario"] == "multiple_sets_rests" for item in report["benchmark_plan"]))

    def test_performance_budget_cannot_be_enforced_without_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            args = SimpleNamespace(
                enforce_budget=True,
                adapter_executable=None,
                stage_report=[],
                output=str(Path(temp) / "performance.json"),
                repetitions=1,
            )
            with self.assertRaises(module.ValidationError):
                module.performance(args)

    def test_legacy_stage_summary_separates_pose_from_post_pose(self) -> None:
        report = {
            "run_id": "unit",
            "total_duration_ms": 1000,
            "process_peak_rss_bytes": 42,
            "stages": [
                {"name": "video_pose_source", "duration_ms": 300},
                {"name": "video_pose_reference", "duration_ms": 400},
                {"name": "motion_source", "duration_ms": 50},
                {"name": "phase_source", "duration_ms": 25},
                {"name": "motion_reference", "duration_ms": 50},
                {"name": "phase_reference", "duration_ms": 25},
                {"name": "alignment", "duration_ms": 100},
            ],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "stage.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            summary = module._legacy_stage_summary(path)
        self.assertEqual(summary["pose_preprocessing_ms"], 700)
        self.assertEqual(summary["post_pose_motion_phase_alignment_ms"], 250)
        self.assertEqual(summary["alignment_only_ms"], 100)
        self.assertAlmostEqual(summary["alignment_fraction_of_total"], 0.1)

    @staticmethod
    def base_result() -> dict:
        return {
            "status": "SYNCHRONIZED",
            "confidence": 0.9,
            "source_analyzable_fraction": 0.9,
            "reference_analyzable_fraction": 0.9,
            "mappings": [
                {
                    "source_ms": 100,
                    "reference_ms": 120,
                    "confidence": 0.9,
                    "source_unit_id": "s0",
                    "reference_unit_id": "r0",
                    "source_direction": "RAISING",
                    "reference_direction": "RAISING",
                    "source_state": "ACTIVE",
                    "reference_state": "ACTIVE",
                    "reliability": "RELIABLE",
                },
                {
                    "source_ms": 200,
                    "reference_ms": 220,
                    "confidence": 0.9,
                    "source_unit_id": "s0",
                    "reference_unit_id": "r0",
                    "source_direction": "LOWERING",
                    "reference_direction": "LOWERING",
                    "source_state": "ACTIVE",
                    "reference_state": "ACTIVE",
                    "reliability": "RELIABLE",
                },
            ],
            "unmatched_source_units": [],
            "unmatched_reference_units": [],
            "spatial_diagnostics": {"mirror": "NOT_MIRRORED"},
        }


if __name__ == "__main__":
    unittest.main()
