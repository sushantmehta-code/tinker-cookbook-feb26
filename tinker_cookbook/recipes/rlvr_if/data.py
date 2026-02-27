import json
import os
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, TypeAlias

import chz
from tinker_cookbook.renderers import Message

Conversation: TypeAlias = list[Message]

_PROMPT_KEYS = ("prompt", "instruction", "input", "question")
_RUBRIC_KEYS = ("rubrics", "rubric_items", "criteria", "requirements", "checks")
_RUBRIC_TEXT_KEYS = (
    "rubric",
    "rubric_str",
    "text",
    "criterion",
    "requirement",
    "description",
    "instruction",
    "content",
)
_SAMPLE_ID_KEYS = ("id", "sample_id", "uid", "uuid")
_WIDE_RUBRIC_CRITERION_KEY_RE = re.compile(r"^rubric\s*-\s*(\d+)\.\s*criterion$", re.IGNORECASE)


def _error(index: int, message: str) -> ValueError:
    return ValueError(f"Invalid datapoint at index {index}: {message}")


def _normalize_text(value: Any, *, index: int, field_name: str) -> str:
    if not isinstance(value, str):
        raise _error(index, f"`{field_name}` must be a string, got {type(value).__name__}")
    out = value.strip()
    if not out:
        raise _error(index, f"`{field_name}` cannot be empty")
    return out


def _normalize_message(value: Any, *, index: int, field_name: str) -> Message:
    if not isinstance(value, Mapping):
        raise _error(index, f"`{field_name}` must contain objects with `role` and `content`")
    role = _normalize_text(value.get("role"), index=index, field_name=f"{field_name}.role")
    content = _normalize_text(value.get("content"), index=index, field_name=f"{field_name}.content")
    return {"role": role, "content": content}


def _normalize_conversation(value: Any, *, index: int, field_name: str) -> list[Message]:
    if not isinstance(value, list):
        raise _error(index, f"`{field_name}` must be a list")
    return [
        _normalize_message(message, index=index, field_name=f"{field_name}[{message_idx}]")
        for message_idx, message in enumerate(value)
    ]


def _extract_rubric_text(value: Any, *, index: int, rubric_index: int) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _error(index, f"`rubrics[{rubric_index}]` cannot be empty")
        return text
    if isinstance(value, Mapping):
        for key in _RUBRIC_TEXT_KEYS:
            if key in value and isinstance(value[key], str) and value[key].strip():
                return value[key].strip()
        raise _error(
            index,
            f"`rubrics[{rubric_index}]` object must include one of {list(_RUBRIC_TEXT_KEYS)}",
        )
    raise _error(
        index,
        f"`rubrics[{rubric_index}]` must be either a string or object, got {type(value).__name__}",
    )


def _extract_rubrics(value: Any, *, index: int) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise _error(index, "`rubrics` cannot be empty")
        return (text,)
    if not isinstance(value, list):
        raise _error(index, "`rubrics` must be a list of rubric strings/objects")

    rubrics = tuple(
        _extract_rubric_text(item, index=index, rubric_index=rubric_idx)
        for rubric_idx, item in enumerate(value)
    )
    if not rubrics:
        raise _error(index, "`rubrics` must contain at least one rubric")
    return rubrics


def _extract_rubrics_from_wide_schema(
    mapping: Mapping[str, Any], *, index: int
) -> tuple[str, ...] | None:
    rubric_pairs: list[tuple[int, Any]] = []
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        key_match = _WIDE_RUBRIC_CRITERION_KEY_RE.match(key.strip())
        if key_match is None:
            continue
        rubric_pairs.append((int(key_match.group(1)), value))

    if not rubric_pairs:
        return None

    rubric_pairs.sort(key=lambda x: x[0])
    rubric_texts: list[str] = []
    for _rubric_idx, rubric_value in rubric_pairs:
        if isinstance(rubric_value, str):
            rubric_text = rubric_value.strip()
            if rubric_text:
                rubric_texts.append(rubric_text)

    if not rubric_texts:
        raise _error(
            index,
            "found `rubric - N. criterion` columns but none contained non-empty rubric text",
        )
    return tuple(rubric_texts)


def _get_first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _find_last_user_message_idx(conversation: Sequence[Message]) -> int | None:
    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx]["role"] == "user":
            return idx
    return None


@dataclass(frozen=True)
class RLVRIfDatapoint:
    prompt: str
    rubric_items: tuple[str, ...]
    convo_prefix: tuple[Message, ...] = ()
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "rubric_items": list(self.rubric_items),
            "convo_prefix": list(self.convo_prefix),
            "sample_id": self.sample_id,
        }

    @staticmethod
    def from_dict(value: Any, *, index: int) -> "RLVRIfDatapoint":
        if not isinstance(value, Mapping):
            raise _error(index, f"expected object, got {type(value).__name__}")

        prompt_raw = _get_first_present(value, _PROMPT_KEYS)
        convo_prefix: tuple[Message, ...] = ()

        convo_prefix_raw = value.get("convo_prefix")
        if convo_prefix_raw is not None:
            convo_prefix = tuple(
                _normalize_conversation(convo_prefix_raw, index=index, field_name="convo_prefix")
            )

        convo_raw = value.get("convo")
        if convo_raw is not None:
            convo = _normalize_conversation(convo_raw, index=index, field_name="convo")
            if prompt_raw is None:
                user_idx = _find_last_user_message_idx(convo)
                if user_idx is None:
                    raise _error(index, "`convo` must contain at least one user message")
                prompt_raw = convo[user_idx]["content"]
                convo_prefix = tuple(convo[:user_idx])
            elif convo_prefix_raw is None:
                prompt_str = _normalize_text(prompt_raw, index=index, field_name="prompt")
                if convo and convo[-1]["role"] == "user" and convo[-1]["content"] == prompt_str:
                    convo_prefix = tuple(convo[:-1])
                else:
                    convo_prefix = tuple(convo)

        if prompt_raw is None:
            raise _error(index, f"missing prompt field; expected one of {list(_PROMPT_KEYS)}")
        prompt = _normalize_text(prompt_raw, index=index, field_name="prompt")

        rubrics_raw = _get_first_present(value, _RUBRIC_KEYS)
        if rubrics_raw is None and "rubric" in value:
            rubrics_raw = [value["rubric"]]
        if rubrics_raw is None:
            wide_schema_rubrics = _extract_rubrics_from_wide_schema(value, index=index)
            if wide_schema_rubrics is None:
                raise _error(index, f"missing rubrics field; expected one of {list(_RUBRIC_KEYS)}")
            rubric_items = wide_schema_rubrics
        else:
            rubric_items = _extract_rubrics(rubrics_raw, index=index)

        sample_id = None
        sample_id_raw = _get_first_present(value, _SAMPLE_ID_KEYS)
        if sample_id_raw is not None:
            sample_id = _normalize_text(sample_id_raw, index=index, field_name="sample_id")

        return RLVRIfDatapoint(
            prompt=prompt,
            rubric_items=rubric_items,
            convo_prefix=convo_prefix,
            sample_id=sample_id,
        )


@chz.chz
class RLVRIfDatapointListBuilder:
    def __call__(self) -> Sequence[RLVRIfDatapoint]:
        raise NotImplementedError("Subclass must implement this method")


def _looks_like_single_datapoint(value: Mapping[str, Any]) -> bool:
    has_prompt = any(key in value for key in _PROMPT_KEYS) or "convo" in value
    has_rubric = any(key in value for key in _RUBRIC_KEYS) or "rubric" in value
    return has_prompt and has_rubric


def _extract_records_from_mapping(
    payload: Mapping[str, Any], *, split: str | None, data_path: str
) -> list[Any]:
    if split is not None:
        split_records = payload.get(split)
        if not isinstance(split_records, list):
            raise ValueError(
                f"Dataset `{data_path}` does not contain list split `{split}`. "
                f"Available top-level keys: {sorted(payload.keys())}"
            )
        return split_records

    for key in ("data", "samples", "examples"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return candidate

    if "train" in payload and "test" in payload:
        raise ValueError(
            f"Dataset `{data_path}` contains both `train` and `test` splits; "
            "set `split` explicitly."
        )

    for key in ("train", "test"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            return candidate

    if _looks_like_single_datapoint(payload):
        return [payload]

    raise ValueError(
        f"Unsupported dataset structure in `{data_path}`. Expected a list of datapoints "
        "or a dict with split/data keys."
    )


@chz.chz
class RLVRIfDatapointListBuilderFromJson(RLVRIfDatapointListBuilder):
    data_path: str
    split: str | None = None
    max_datapoints: int | None = None

    def _load_records(self) -> list[Any]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "Copy your dataset into `tinker_cookbook/recipes/rlvr_if/` and set "
                "`train_data_path`/`test_data_path` to that file."
            )

        if self.data_path.endswith(".jsonl"):
            records: list[Any] = []
            with open(self.data_path, "r") as f:
                for line_idx, line in enumerate(f, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        records.append(json.loads(stripped))
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in `{self.data_path}` line {line_idx}: {exc}"
                        ) from exc
            return records

        with open(self.data_path, "r") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            return payload
        if isinstance(payload, Mapping):
            return _extract_records_from_mapping(payload, split=self.split, data_path=self.data_path)
        raise ValueError(
            f"Unsupported JSON top-level type in `{self.data_path}`: {type(payload).__name__}"
        )

    def __call__(self) -> Sequence[RLVRIfDatapoint]:
        records = self._load_records()
        if self.max_datapoints is not None:
            if self.max_datapoints <= 0:
                raise ValueError("`max_datapoints` must be positive when provided")
            records = records[: self.max_datapoints]

        datapoints = [
            RLVRIfDatapoint.from_dict(record, index=i) for i, record in enumerate(records)
        ]
        if not datapoints:
            raise ValueError(
                f"No datapoints loaded from `{self.data_path}` "
                f"(split={self.split}, max_datapoints={self.max_datapoints})"
            )
        return datapoints
