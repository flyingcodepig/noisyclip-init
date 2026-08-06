"""Unit tests for official prediction CSV writing."""

from __future__ import annotations

import pytest

from noisyclip.submission.mapping import validate_class_mapping
from noisyclip.submission.writer import PredictionWriteError, write_prediction_csv


def test_writer_outputs_headerless_four_digit_csv(tmp_path):
    """Writer emits exactly `filename,class_id` rows without a header."""

    mapping = validate_class_mapping({"0001": 0, "0010": 1})
    output = tmp_path / "pred_results.csv"

    written = write_prediction_csv(["A.JPG", "b.png"], [1, 0], mapping, output)

    assert written == output
    assert output.read_text(encoding="utf-8") == "A.JPG,0010\nb.png,0001\n"


@pytest.mark.parametrize(
    "filenames, predictions, message",
    [
        (["a.jpg"], [0, 1], "differs"),
        (["a.jpg", "a.jpg"], [0, 1], "duplicate"),
        ([""], [0], "non-empty"),
        (["a.jpg"], [2], "outside"),
    ],
)
def test_writer_rejects_bad_inputs(tmp_path, filenames, predictions, message):
    """Length, duplicate filename, empty filename, and invalid index errors fail."""

    mapping = validate_class_mapping({"0001": 0, "0002": 1})

    with pytest.raises((PredictionWriteError, ValueError), match=message):
        write_prediction_csv(filenames, predictions, mapping, tmp_path / "pred_results.csv")


def test_writer_requires_official_name_and_refuses_existing_output(tmp_path):
    """Destination must be `pred_results.csv` and cannot be overwritten by default."""

    mapping = validate_class_mapping({"0001": 0})
    existing = tmp_path / "pred_results.csv"
    existing.write_text("old,0001\n", encoding="utf-8")

    with pytest.raises(PredictionWriteError, match="named"):
        write_prediction_csv(["a.jpg"], [0], mapping, tmp_path / "other.csv")
    with pytest.raises(PredictionWriteError, match="overwrite"):
        write_prediction_csv(["a.jpg"], [0], mapping, existing)
