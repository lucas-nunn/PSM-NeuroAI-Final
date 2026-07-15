import tempfile
import unittest
from contextlib import ExitStack
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from psm_final.analysis import runner


def _encoding_frame(index_name, labels, scores=None):
    labels = list(labels)
    scores = list(scores if scores is not None else [0.4] * len(labels))
    return pd.DataFrame(
        {
            "mean_encoding_score": scores,
            "std_encoding_score": [0.05] * len(labels),
            "best_alpha": [0.1] * len(labels),
            "alpha_selection_stability": [0.8] * len(labels),
            "outer_selected_alphas": ["0.1,0.1,0.01,0.1,0.1"] * len(labels),
            "n_targets": [12] * len(labels),
            "n_outer_folds": [5] * len(labels),
        },
        index=pd.Index(labels, name=index_name),
    )


class _FakeEncodingAnalyzer:
    def __init__(
        self, algonauts_table=None, triple_n_table=None, invoke_progress=False
    ):
        self.algonauts_table = (
            _encoding_frame("roi", ["V1"])
            if algonauts_table is None
            else algonauts_table
        )
        self.triple_n_table = (
            _encoding_frame("area_label", ["IT"])
            if triple_n_table is None
            else triple_n_table
        )
        self.invoke_progress = invoke_progress
        self.calls = []

    def encoding_tables(self, algonauts, triple_n, shared_ids, **kwargs):
        self.calls.append((algonauts, triple_n, shared_ids, kwargs))
        if self.invoke_progress:
            kwargs["progress"]("algonauts", "V1", 1, 1)
        return {
            "algonauts": self.algonauts_table.copy(),
            "triple_n": self.triple_n_table.copy(),
        }


class RunnerParserAndFilteringTests(unittest.TestCase):
    def test_default_method_preserves_rsa_behavior(self):
        args = runner.build_parser().parse_args([])

        self.assertEqual(args.method, "rsa")
        self.assertEqual(args.regression, "ridge")
        self.assertEqual(args.outer_folds, 5)
        self.assertEqual(args.inner_folds, 3)
        self.assertEqual(args.seed, 42)
        self.assertIsNone(args.output_dir)

    def test_encoding_cli_options_parse(self):
        args = runner.build_parser().parse_args(
            [
                "--method",
                "encoding",
                "--regression",
                "lasso",
                "--alphas",
                "0.01",
                "1.0",
                "--outer-folds",
                "4",
                "--inner-folds",
                "2",
                "--seed",
                "7",
                "--models",
                "*vae*",
                "pixel*",
                "--resume",
            ]
        )

        self.assertEqual(args.method, "encoding")
        self.assertEqual(args.regression, "lasso")
        self.assertEqual(args.alphas, [0.01, 1.0])
        self.assertEqual(args.outer_folds, 4)
        self.assertEqual(args.inner_folds, 2)
        self.assertEqual(args.seed, 7)
        self.assertEqual(args.models, ["*vae*", "pixel*"])
        self.assertTrue(args.resume)

    def test_model_patterns_are_case_insensitive_deduplicated_and_ordered(self):
        factories = [Mock(name=f"factory_{i}") for i in range(3)]
        specs = [
            ("beta-VAE z=64", factories[0]),
            ("VDVAE res<=16", factories[1]),
            ("Pixel baseline", factories[2]),
        ]

        selected = runner.filter_model_specs(specs, ["*VAE*", "beta-*"])

        self.assertEqual(selected, specs[:2])

    def test_no_model_patterns_preserves_all_specs(self):
        specs = [("one", Mock()), ("two", Mock())]
        self.assertEqual(runner.filter_model_specs(specs), specs)

    def test_model_output_ids_disambiguate_slug_collisions(self):
        specs = [("A/B", Mock()), ("A B", Mock()), ("Unique", Mock())]

        output_ids = runner._model_output_ids(specs)
        ids = (
            list(output_ids.values())
            if isinstance(output_ids, dict)
            else list(output_ids)
        )

        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(value for value in ids))
        self.assertEqual(
            output_ids["A/B"], runner._model_output_ids([specs[0]])["A/B"]
        )


class RunEncodingTests(unittest.TestCase):
    def setUp(self):
        self.algonauts = object()
        self.triple_n = object()
        self.shared_ids = list(range(101, 111))

    def _run(self, model_specs, output_dir=None, **overrides):
        options = {
            "subjects": [1, 2],
            "rois": ["V1"],
            "area_labels": ["IT"],
            "regression": "ridge",
            "alphas": [0.01, 0.1],
            "outer_folds": 5,
            "inner_folds": 3,
            "seed": 9,
            "output_dir": output_dir,
            "resume": False,
            "fail_fast": False,
        }
        options.update(overrides)
        with patch.object(
            runner.ModelAnalysisBase,
            "aligned_stimuli",
            return_value=(
                list(range(101, 111)),
                list(range(1, 11)),
            ),
        ):
            return runner.run_encoding(
                self.algonauts,
                self.triple_n,
                self.shared_ids,
                model_specs,
                **options,
            )

    def test_forwards_encoding_configuration_and_aggregates_model_tables(self):
        analyzer = _FakeEncodingAnalyzer()
        factory = Mock(return_value=analyzer)

        results = self._run([("Demo Model", factory)])

        factory.assert_called_once_with()
        self.assertEqual(list(results["algonauts"]), ["Demo Model"])
        self.assertEqual(list(results["triple_n"]), ["Demo Model"])
        self.assertEqual(len(analyzer.calls), 1)
        algonauts, triple_n, shared_ids, kwargs = analyzer.calls[0]
        self.assertIs(algonauts, self.algonauts)
        self.assertIs(triple_n, self.triple_n)
        self.assertIs(shared_ids, self.shared_ids)
        self.assertEqual(kwargs["subjects"], [1, 2])
        self.assertEqual(kwargs["rois"], ["V1"])
        self.assertEqual(kwargs["area_labels"], ["IT"])
        self.assertEqual(kwargs["regression"], "ridge")
        self.assertEqual(kwargs["alphas"], [0.01, 0.1])
        self.assertEqual(kwargs["outer_folds"], 5)
        self.assertEqual(kwargs["inner_folds"], 3)
        self.assertEqual(kwargs["seed"], 9)

    def test_progress_callback_identifies_model_modality_group_and_count(self):
        analyzer = _FakeEncodingAnalyzer(invoke_progress=True)

        stream = StringIO()
        with redirect_stdout(stream):
            self._run([("Demo Model", Mock(return_value=analyzer))])

        self.assertIn(
            "[encoding] Demo Model: algonauts V1 [1/1]", stream.getvalue()
        )

    def test_resume_loads_a_complete_same_configuration_without_factory(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = _FakeEncodingAnalyzer()
            self._run([("Demo Model", Mock(return_value=first))], output_dir=tmp)
            forbidden_factory = Mock(
                side_effect=AssertionError("completed model should not be loaded")
            )

            resumed = self._run(
                [("Demo Model", forbidden_factory)],
                output_dir=tmp,
                resume=True,
            )

            forbidden_factory.assert_not_called()
            self.assertIn("Demo Model", resumed["algonauts"])
            self.assertIn("Demo Model", resumed["triple_n"])

    def test_resume_reruns_a_partial_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(
                [("Demo Model", Mock(return_value=_FakeEncodingAnalyzer()))],
                output_dir=tmp,
            )
            triple_n_files = [
                path
                for path in Path(tmp).rglob("*triple_n*.csv")
                if "_all" not in path.name
            ]
            self.assertTrue(triple_n_files)
            for path in triple_n_files:
                path.unlink()
            replacement = _FakeEncodingAnalyzer()
            replacement_factory = Mock(return_value=replacement)

            self._run(
                [("Demo Model", replacement_factory)],
                output_dir=tmp,
                resume=True,
            )

            replacement_factory.assert_called_once_with()
            self.assertEqual(len(replacement.calls), 1)

    def test_resume_does_not_reuse_a_different_alpha_sweep(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(
                [("Demo Model", Mock(return_value=_FakeEncodingAnalyzer()))],
                output_dir=tmp,
                alphas=[0.01],
            )
            replacement = _FakeEncodingAnalyzer()
            replacement_factory = Mock(return_value=replacement)

            self._run(
                [("Demo Model", replacement_factory)],
                output_dir=tmp,
                resume=True,
                alphas=[1.0],
            )

            replacement_factory.assert_called_once_with()

    def test_resume_rejects_a_parseable_but_modified_result_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(
                [("Demo Model", Mock(return_value=_FakeEncodingAnalyzer()))],
                output_dir=tmp,
            )
            result_csv = next(Path(tmp).glob("*_algonauts.csv"))
            result_csv.write_text(
                result_csv.read_text().replace("0.4", "0.9", 1)
            )
            replacement = _FakeEncodingAnalyzer()
            replacement_factory = Mock(return_value=replacement)

            self._run(
                [("Demo Model", replacement_factory)],
                output_dir=tmp,
                resume=True,
            )

            replacement_factory.assert_called_once_with()

    def test_resume_rejects_changed_model_artifact_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "weights.bin"
            checkpoint.write_bytes(b"first weights")
            holder = {"analyzer": _FakeEncodingAnalyzer()}

            def factory(path=checkpoint):
                self.assertTrue(path.exists())
                return holder["analyzer"]

            output = Path(tmp) / "results"
            self._run([("Demo Model", factory)], output_dir=output)
            replacement = _FakeEncodingAnalyzer()
            holder["analyzer"] = replacement
            checkpoint.write_bytes(b"different weights")

            self._run(
                [("Demo Model", factory)],
                output_dir=output,
                resume=True,
            )

            self.assertEqual(len(replacement.calls), 1)

    def test_new_run_invalidates_stale_same_regression_heatmaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            stale = [
                Path(tmp) / "encoding_ridge_algonauts.png",
                Path(tmp) / "encoding_ridge_triple_n_area_label.png",
            ]
            for path in stale:
                path.write_bytes(b"old plot")

            self._run(
                [("Demo Model", Mock(return_value=_FakeEncodingAnalyzer()))],
                output_dir=tmp,
            )

            self.assertTrue(all(not path.exists() for path in stale))

    def test_known_unsafe_factory_cannot_bypass_safety_via_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            factory = Mock(return_value=_FakeEncodingAnalyzer())
            self._run([("Demo Model", factory)], output_dir=tmp)
            factory.encoding_cv_safe = False

            with self.assertWarnsRegex(UserWarning, "data leakage"):
                results = self._run(
                    [("Demo Model", factory)],
                    output_dir=tmp,
                    resume=True,
                )

            self.assertEqual(factory.call_count, 1)
            self.assertIn("Demo Model", results["skipped"])
            self.assertNotIn("Demo Model", results["algonauts"])
            self.assertEqual(list(Path(tmp).glob("*_algonauts.csv")), [])

    def test_impossible_fold_counts_fail_before_model_construction(self):
        factory = Mock(return_value=_FakeEncodingAnalyzer())
        with patch.object(
            runner.ModelAnalysisBase,
            "aligned_stimuli",
            return_value=([101, 102, 103], [1, 2, 3]),
        ):
            with self.assertRaisesRegex(ValueError, "outer CV"):
                runner.run_encoding(
                    self.algonauts,
                    self.triple_n,
                    self.shared_ids,
                    [("Demo Model", factory)],
                    subjects=[1],
                    rois=["V1"],
                    area_labels=["IT"],
                    outer_folds=5,
                    inner_folds=3,
                )

        factory.assert_not_called()

    def test_failure_isolated_and_later_model_still_runs(self):
        failed_factory = Mock(side_effect=RuntimeError("broken checkpoint"))
        later = _FakeEncodingAnalyzer()

        with self.assertWarnsRegex(UserWarning, "broken checkpoint"):
            results = self._run(
                [
                    ("Broken", failed_factory),
                    ("Healthy", Mock(return_value=later)),
                ]
            )

        self.assertNotIn("Broken", results["algonauts"])
        self.assertIn("Healthy", results["algonauts"])
        self.assertEqual(len(later.calls), 1)

    def test_fail_fast_propagates_model_failure(self):
        with self.assertWarnsRegex(UserWarning, "broken checkpoint"):
            with self.assertRaisesRegex(RuntimeError, "broken checkpoint"):
                self._run(
                    [
                        (
                            "Broken",
                            Mock(side_effect=RuntimeError("broken checkpoint")),
                        )
                    ],
                    fail_fast=True,
                )


class EncodingOutputTests(unittest.TestCase):
    def _results(self, regression="ridge"):
        return {
            "algonauts": {
                "Model A": _encoding_frame("roi", ["V1", "V2"], [0.2, 0.5]),
                "Model B": _encoding_frame("roi", ["V1", "V2"], [0.3, 0.4]),
            },
            "triple_n": {
                "Model A": _encoding_frame("area_label", ["IT"], [0.6]),
                "Model B": _encoding_frame("area_label", ["IT"], [0.7]),
            },
            "config": {
                "regression": regression,
                "alphas": [0.01, 0.1],
                "outer_folds": 5,
                "inner_folds": 3,
                "seed": 42,
            },
            "subjects": [1, 2],
            "errors": {},
        }

    def test_saves_per_model_and_combined_tidy_csvs_without_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            combined = runner.save_encoding_results(
                self._results(), tmp, make_plots=False
            )

            self.assertEqual(Path(combined).name, "encoding_ridge_all.csv")
            long = pd.read_csv(combined)
            self.assertEqual(len(long), 6)
            self.assertTrue(
                {
                    "model",
                    "modality",
                    "segmentation",
                    "region",
                    "mean_encoding_score",
                    "best_alpha",
                }.issubset(long.columns)
            )
            self.assertEqual(set(long["modality"]), {"algonauts", "triple_n"})
            per_model = [
                path.name
                for path in Path(tmp).glob("*.csv")
                if path != Path(combined)
            ]
            self.assertTrue(
                any("ridge" in name and "algonauts" in name for name in per_model)
            )
            self.assertTrue(
                any("ridge" in name and "triple_n" in name for name in per_model)
            )
            self.assertEqual(list(Path(tmp).glob("*.png")), [])

    def test_writes_two_nonempty_heatmaps_and_closes_figures(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            runner.save_encoding_results(self._results(), tmp, make_plots=True)

            pngs = list(Path(tmp).glob("*.png"))
            self.assertEqual(len(pngs), 2)
            self.assertTrue(all(path.stat().st_size > 0 for path in pngs))
            self.assertEqual(plt.get_fignums(), [])


class RunnerMainRoutingTests(unittest.TestCase):
    def _patch_common(self):
        rsa_results = {
            "algonauts": {},
            "triple_n": {},
            "n_stimuli": 0,
            "subjects": [1],
        }
        stack = ExitStack()
        mocks = {
            "load_dotenv": stack.enter_context(patch.object(runner, "_load_dotenv")),
            "shared_stimuli": stack.enter_context(
                patch.object(runner, "shared_stimuli", return_value=[1])
            ),
            "algonauts": stack.enter_context(
                patch.object(runner, "Algonauts", return_value=object())
            ),
            "triple_n": stack.enter_context(
                patch.object(runner, "TripleN", return_value=object())
            ),
            "discover": stack.enter_context(
                patch.object(runner, "discover_analyzers", return_value=[])
            ),
            "collect": stack.enter_context(
                patch.object(runner, "collect_models", return_value=[("Demo", Mock())])
            ),
            "run_rsa": stack.enter_context(
                patch.object(runner, "run_rsa", return_value=rsa_results)
            ),
            "run_encoding": stack.enter_context(
                patch.object(
                    runner,
                    "run_encoding",
                    return_value={
                        "algonauts": {},
                        "triple_n": {},
                        "config": {"regression": "ridge"},
                        "errors": {},
                    },
                )
            ),
            "save_rsa": stack.enter_context(patch.object(runner, "save_results")),
            "save_encoding": stack.enter_context(
                patch.object(runner, "save_encoding_results")
            ),
            "summary": stack.enter_context(patch.object(runner, "print_summary")),
            "encoding_summary": stack.enter_context(
                patch.object(runner, "print_encoding_summary")
            ),
        }
        return stack, mocks

    def test_default_main_dispatches_only_rsa_to_legacy_output_directory(self):
        stack, mocks = self._patch_common()
        with stack:
            runner.main(
                [
                    "--algonauts-dir",
                    "algo",
                    "--triple-n-dir",
                    "tn",
                    "--no-plots",
                ]
            )

        mocks["run_rsa"].assert_called_once()
        mocks["run_encoding"].assert_not_called()
        mocks["save_rsa"].assert_called_once()
        mocks["save_encoding"].assert_not_called()
        self.assertEqual(
            Path(mocks["save_rsa"].call_args.args[1]), runner.DEFAULT_OUTPUT_DIR
        )

    def test_encoding_main_dispatches_only_encoding(self):
        stack, mocks = self._patch_common()
        with stack:
            runner.main(
                [
                    "--method",
                    "encoding",
                    "--algonauts-dir",
                    "algo",
                    "--triple-n-dir",
                    "tn",
                    "--no-plots",
                ]
            )

        mocks["run_rsa"].assert_not_called()
        mocks["run_encoding"].assert_called_once()
        mocks["save_rsa"].assert_not_called()
        mocks["save_encoding"].assert_called_once()
        self.assertEqual(
            Path(mocks["save_encoding"].call_args.args[1]),
            runner.DEFAULT_ENCODING_OUTPUT_DIR,
        )

    def test_default_encoding_batch_excludes_opt_in_resource_heavy_models(self):
        stack, mocks = self._patch_common()
        heavy_factory = Mock()
        heavy_factory.encoding_default = False
        normal_factory = Mock()
        mocks["collect"].return_value = [
            ("Pixel", heavy_factory),
            ("Demo", normal_factory),
        ]

        with stack:
            runner.main(
                [
                    "--method",
                    "encoding",
                    "--algonauts-dir",
                    "algo",
                    "--triple-n-dir",
                    "tn",
                    "--no-plots",
                ]
            )

        selected_specs = mocks["run_encoding"].call_args.args[3]
        self.assertEqual([label for label, _factory in selected_specs], ["Demo"])

    def test_partial_encoding_failure_returns_nonzero_after_saving_outputs(self):
        stack, mocks = self._patch_common()
        mocks["run_encoding"].return_value = {
            "algonauts": {"Healthy": object()},
            "triple_n": {"Healthy": object()},
            "config": {"regression": "ridge"},
            "errors": {"Broken": "RuntimeError: failed"},
            "skipped": {},
        }

        with stack:
            with self.assertRaisesRegex(SystemExit, "1 encoding model"):
                runner.main(
                    [
                        "--method",
                        "encoding",
                        "--algonauts-dir",
                        "algo",
                        "--triple-n-dir",
                        "tn",
                        "--no-plots",
                    ]
                )

        mocks["save_encoding"].assert_called_once()


if __name__ == "__main__":
    unittest.main()
