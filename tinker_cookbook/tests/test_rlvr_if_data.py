import json

import pytest

from tinker_cookbook.recipes.rlvr_if.data import RLVRIfDatapointListBuilderFromJson


def test_load_minimal_datapoint_with_alias_fields(tmp_path):
    payload = [
        {
            "instruction": "Write one sentence about apples.",
            "criteria": [{"text": "Mentions apples"}, "Exactly one sentence"],
            "id": "sample-001",
        }
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload))

    datapoints = RLVRIfDatapointListBuilderFromJson(data_path=str(path))()

    assert len(datapoints) == 1
    assert datapoints[0].prompt == "Write one sentence about apples."
    assert datapoints[0].rubric_items == ("Mentions apples", "Exactly one sentence")
    assert datapoints[0].sample_id == "sample-001"


def test_extract_prompt_and_prefix_from_conversation(tmp_path):
    payload = [
        {
            "convo": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "Answer one"},
                {"role": "user", "content": "Final question"},
            ],
            "rubrics": ["Answers the final question"],
        }
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload))

    datapoints = RLVRIfDatapointListBuilderFromJson(data_path=str(path))()

    assert datapoints[0].prompt == "Final question"
    assert len(datapoints[0].convo_prefix) == 3
    assert datapoints[0].convo_prefix[-1]["role"] == "assistant"


def test_split_dict_requires_explicit_split(tmp_path):
    payload = {
        "train": [{"prompt": "p1", "rubrics": ["r1"]}],
        "test": [{"prompt": "p2", "rubrics": ["r2"]}],
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="set `split` explicitly"):
        RLVRIfDatapointListBuilderFromJson(data_path=str(path))()

    train_datapoints = RLVRIfDatapointListBuilderFromJson(data_path=str(path), split="train")()
    test_datapoints = RLVRIfDatapointListBuilderFromJson(data_path=str(path), split="test")()

    assert len(train_datapoints) == 1
    assert len(test_datapoints) == 1
    assert train_datapoints[0].prompt == "p1"
    assert test_datapoints[0].prompt == "p2"


def test_validation_error_includes_index(tmp_path):
    payload = [
        {"prompt": "ok", "rubrics": ["r"]},
        {"prompt": "missing rubrics"},
    ]
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="index 1"):
        RLVRIfDatapointListBuilderFromJson(data_path=str(path))()


def test_jsonl_invalid_line_reports_line_number(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"prompt":"ok","rubrics":["r"]}\n{not valid json}\n')

    with pytest.raises(ValueError, match="line 2"):
        RLVRIfDatapointListBuilderFromJson(data_path=str(path))()


def test_wide_schema_rubric_columns_are_sorted_numerically(tmp_path):
    payload = [
        {
            "task_id": "t1",
            "task_response_id": "r1",
            "prompt": "Draft a short update",
            "rubric - 10. criterion": "Mentions next steps",
            "rubric - 2. criterion": "Uses a professional tone",
            "rubric - 1. criterion": "Includes a greeting",
            "rubric - 2. criterion_origin": "extra metadata",
        }
    ]
    path = tmp_path / "wide_schema.jsonl"
    path.write_text(json.dumps(payload[0]) + "\n")

    datapoints = RLVRIfDatapointListBuilderFromJson(data_path=str(path))()

    assert len(datapoints) == 1
    assert datapoints[0].rubric_items == (
        "Includes a greeting",
        "Uses a professional tone",
        "Mentions next steps",
    )


def test_wide_schema_ignores_empty_rubric_cells(tmp_path):
    payload = [
        {
            "prompt": "Write one paragraph.",
            "rubric - 1. criterion": "   ",
            "rubric - 2. criterion": "Has one paragraph",
            "rubric - 3. criterion": None,
            "rubric - 4. criterion": 123,
        }
    ]
    path = tmp_path / "wide_schema_sparse.json"
    path.write_text(json.dumps(payload))

    datapoints = RLVRIfDatapointListBuilderFromJson(data_path=str(path))()

    assert len(datapoints) == 1
    assert datapoints[0].rubric_items == ("Has one paragraph",)
