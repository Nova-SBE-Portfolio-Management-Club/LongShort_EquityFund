import json

import numpy as np
import pandas as pd
import pytest

from scripts.reproduce_results import (
    check_saved,
    manifest_for,
    plot_results,
    report_text,
    sha256,
    synthetic_results,
)
from src.playground.backtest_demo import generate_sample_data


def test_demo_rng_is_reproducible_and_does_not_reset_global_state():
    np.random.seed(123)
    expected_next = np.random.random()
    np.random.seed(123)
    first = generate_sample_data()
    second = generate_sample_data()
    assert np.random.random() == expected_next
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_series_equal(left, right)


def saved_example(tmp_path):
    name = "synthetic_demo"
    frame, config, inputs, audit = synthetic_results()
    manifest = manifest_for(frame, config, inputs, audit)
    (tmp_path / "figures").mkdir()
    frame.to_csv(tmp_path / f"{name}.returns.csv", index_label="date", float_format="%.17g")
    (tmp_path / f"{name}.md").write_text(report_text(name, manifest))
    plot_results(frame, "Synthetic example", tmp_path / f"figures/{name}.png")
    manifest["artifact_sha256"] = {
        path: sha256(tmp_path / path)
        for path in (f"{name}.md", f"{name}.returns.csv", f"figures/{name}.png")
    }
    (tmp_path / f"{name}.manifest.json").write_text(json.dumps(manifest, allow_nan=False))
    return name, frame, manifest


def test_reproduction_accepts_consistent_saved_artifacts(tmp_path):
    name, frame, manifest = saved_example(tmp_path)
    check_saved(tmp_path, name, frame, manifest)
    assert manifest["metrics"]["Flat cash"]["Sharpe Ratio"] is None


def test_reproduction_rejects_changed_returns(tmp_path):
    name, frame, manifest = saved_example(tmp_path)
    path = tmp_path / f"{name}.returns.csv"
    altered = pd.read_csv(path, index_col=0)
    altered.iloc[0, 0] += 0.01
    altered.to_csv(path)
    with pytest.raises(AssertionError):
        check_saved(tmp_path, name, frame, manifest)


def test_reproduction_rejects_a_changed_figure(tmp_path):
    name, frame, manifest = saved_example(tmp_path)
    (tmp_path / f"figures/{name}.png").write_bytes(b"changed chart")
    with pytest.raises(ValueError, match="artifact changed"):
        check_saved(tmp_path, name, frame, manifest)


def test_reproduction_rejects_changed_source_manifest(tmp_path):
    name, frame, manifest = saved_example(tmp_path)
    path = tmp_path / f"{name}.manifest.json"
    saved = json.loads(path.read_text())
    saved["source_sha256"]["src/utils/backtest.py"] = "incorrect"
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="source_sha256"):
        check_saved(tmp_path, name, frame, manifest)
