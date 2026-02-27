import asyncio
from types import SimpleNamespace
import warnings

import pytest
import tinker

from tinker_cookbook.recipes.rlvr_if.data import RLVRIfDatapoint
from tinker_cookbook.recipes.rlvr_if.env import (
    OpenAIResponsesMessageCompleter,
    RLVRIfEnv,
    _extract_openai_responses_text,
    _is_openai_gpt5_model,
    build_grader_llm,
    build_judge_prompt,
    parse_judge_response,
)


class SequenceJudge:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = 0

    async def __call__(self, _prompt):
        if self.calls >= len(self.outcomes):
            raise RuntimeError("No more canned judge outcomes")
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return {"role": "assistant", "content": outcome}


class DummyRenderer:
    def parse_response(self, _action):
        return {"role": "assistant", "content": "candidate response"}, True

    def build_generation_prompt(self, _convo):
        return tinker.ModelInput.empty()

    def get_stop_sequences(self):
        return []


def _make_env(judge, *, judge_max_retries=3, judge_min_coverage_ratio=0.85):
    rubrics = tuple(f"rubric-{i}" for i in range(20))
    datapoint = RLVRIfDatapoint(prompt="Do the task.", rubric_items=rubrics)
    return RLVRIfEnv(
        renderer=DummyRenderer(),
        datapoint=datapoint,
        grader_llm=judge,
        judge_max_retries=judge_max_retries,
        judge_min_coverage_ratio=judge_min_coverage_ratio,
    )


def test_parse_strict_json_response():
    response = (
        '{"ratings":[{"rating":"Yes","rationale":"met"},'
        '{"rating":"No","rationale":"missing"}]}'
    )
    result = parse_judge_response(response, num_rubrics=2)

    assert result.pass_rate == 0.5
    assert result.per_rubric_pass == (1.0, 0.0)
    assert result.parse_ok is True
    assert result.used_fallback is False
    assert result.num_rubrics_scored == 2
    assert result.num_rubrics_expected == 2
    assert result.coverage_ratio == 1.0


def test_parse_fenced_json_response():
    response = """
The score is:
```json
{"ratings":[{"rating":"Yes"},{"rating":"No"}]}
```
"""
    result = parse_judge_response(response, num_rubrics=2)

    assert result.pass_rate == 0.5
    assert result.per_rubric_pass == (1.0, 0.0)
    assert result.parse_ok is True
    assert result.used_fallback is True


def test_fallback_regex_parser_handles_malformed_output():
    response = """
rubric results:
"rating": "Yes"
"rating": "No"
"pass_rate": 0.75
"""
    result = parse_judge_response(response, num_rubrics=2)

    # Reward math is computed from scored rubric ratings only.
    assert result.pass_rate == 0.5
    assert result.per_rubric_pass == (1.0, 0.0)
    assert result.parse_ok is True
    assert result.used_fallback is True


def test_parser_clamps_out_of_range_values():
    response = '{"pass_rate": 1.7, "ratings":[{"rating":"Yes"},{"rating":"Yes"}]}'
    result = parse_judge_response(response, num_rubrics=2)

    assert result.pass_rate == 1.0


def test_unparseable_response_defaults_to_zero_reward():
    result = parse_judge_response("not parseable", num_rubrics=3)

    assert result.pass_rate == 0.0
    assert result.per_rubric_pass == ()
    assert result.parse_ok is False
    assert result.used_fallback is True


def test_missing_rubrics_use_scored_only_denominator():
    response = '{"ratings":[{"rating":"Yes"},{"rating":"No"},{"rating":"Yes"}]}'
    result = parse_judge_response(response, num_rubrics=10)

    assert result.per_rubric_pass == (1.0, 0.0, 1.0)
    assert result.pass_rate == 2.0 / 3.0
    assert result.num_rubrics_scored == 3
    assert result.coverage_ratio == 0.3


def test_judge_retry_succeeds_before_max_attempts():
    yes_rating = '{"rating":"Yes","rationale":"met"}'
    good_payload = '{"ratings": [' + ",".join([yes_rating] * 17) + "]}"
    judge = SequenceJudge([RuntimeError("transient"), good_payload])
    env = _make_env(judge)

    parse_result, _latency_ms, attempts, judge_success, failure_reason = asyncio.run(
        env._judge_response("candidate")
    )

    assert judge_success is True
    assert attempts == 2
    assert parse_result.coverage_ratio == 0.85
    assert parse_result.coverage_ok is True
    assert failure_reason is None


def test_judge_retry_exhausted_on_low_coverage():
    yes_rating = '{"rating":"Yes","rationale":"met"}'
    low_coverage_payload = '{"ratings": [' + ",".join([yes_rating] * 16) + "]}"
    judge = SequenceJudge([low_coverage_payload, low_coverage_payload, low_coverage_payload])
    env = _make_env(judge, judge_max_retries=3, judge_min_coverage_ratio=0.85)

    parse_result, _latency_ms, attempts, judge_success, failure_reason = asyncio.run(
        env._judge_response("candidate")
    )

    assert judge_success is False
    assert attempts == 3
    assert parse_result.coverage_ratio == 0.8
    assert parse_result.coverage_ok is False
    assert "below threshold" in (failure_reason or "")


def test_step_marks_rollout_invalid_when_judge_fails():
    judge = SequenceJudge(
        [RuntimeError("judge down"), RuntimeError("judge down"), RuntimeError("judge down")]
    )
    env = _make_env(judge, judge_max_retries=3)
    step_result = asyncio.run(env.step([1, 2, 3]))

    assert step_result.reward == 0.0
    assert step_result.metrics["rollout_valid"] == 0.0
    assert step_result.metrics["judge_call_success"] == 0.0
    assert step_result.metrics["judge_retry_count"] == 2


def test_build_judge_prompt_mentions_expected_count():
    prompt = build_judge_prompt(
        instruction="Do three things.",
        candidate_response="Done.",
        rubrics=("A", "B", "C"),
    )
    content = prompt[0]["content"]

    assert "There are exactly 3 rubric items below. You MUST return exactly 3 ratings." in content
    assert 'Respond as a JSON object with a "ratings" field' in content
    assert '- rating: ONLY "Yes" or "No"' in content
    assert "Prompt:" in content


def test_parse_legacy_per_rubric_still_supported():
    response = '{"per_rubric":[{"pass":1},{"pass":0}],"pass_rate":0.5}'
    result = parse_judge_response(response, num_rubrics=2)

    assert result.pass_rate == 0.5
    assert result.per_rubric_pass == (1.0, 0.0)
    assert result.parse_ok is True


def test_is_openai_gpt5_model_detection():
    assert _is_openai_gpt5_model("gpt-5-mini")
    assert _is_openai_gpt5_model("openai/gpt-5-mini")
    assert not _is_openai_gpt5_model("Qwen/Qwen3-4B-Instruct-2507")


def test_extract_openai_responses_text_prefers_output_text():
    response = SimpleNamespace(output_text="  final answer  ", output=[])
    assert _extract_openai_responses_text(response) == "final answer"


def test_extract_openai_responses_text_fallback_to_output_blocks():
    response = SimpleNamespace(
        output_text="",
        output=[
            SimpleNamespace(content=[SimpleNamespace(text="part-1 "), SimpleNamespace(text="part-2")]),
        ],
    )
    assert _extract_openai_responses_text(response) == "part-1 part-2"


def test_build_grader_llm_gpt5_requires_openai_api_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_grader_llm(
            grader_llm_name="gpt-5-mini",
            policy_renderer_name="qwen3_instruct",
            policy_model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        )


def test_build_grader_llm_routes_gpt5_to_openai(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        grader = build_grader_llm(
            grader_llm_name="gpt-5-mini",
            policy_renderer_name="qwen3_instruct",
            policy_model_name_for_tokenizer="Qwen/Qwen3-4B-Instruct-2507",
        )

    assert isinstance(grader, OpenAIResponsesMessageCompleter)
    assert not any("Could not load tokenizer" in str(w.message) for w in caught)
