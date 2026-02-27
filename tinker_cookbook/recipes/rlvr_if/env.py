import json
import asyncio
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import chz
import tinker
from tinker.types import ModelInput
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import MessageCompleter, StopCondition, TinkerMessageCompleter
from tinker_cookbook.recipes.rlvr_if.data import RLVRIfDatapoint, RLVRIfDatapointListBuilder
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.types import Action, Env, EnvGroupBuilder, RLDataset, RLDatasetBuilder, StepResult
from tinker_cookbook.tokenizer_utils import get_tokenizer

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_PASS_FIELD_RE = re.compile(r'"pass(?:ed)?"\s*:\s*(true|false|1|0)', re.IGNORECASE)
_RATING_FIELD_RE = re.compile(
    r'"rating"\s*:\s*"?(yes|no|true|false|1|0)"?', re.IGNORECASE
)

JUDGE_USER_TEMPLATE = """Prompt:
{prompt}

Completion:
{completion}

**There are exactly {num_rubrics} rubric items below. You MUST return exactly {num_rubrics} ratings.**

Rubric items:
{rubrics}

Respond as a JSON object with a "ratings" field containing a list of EXACTLY {num_rubrics} objects (one for each rubric above), where each object has:
- rating: ONLY "Yes" or "No" (no other values allowed - not "Partial", "Maybe", "N/A", etc.)
- rationale: str briefly explaining the rating

**STRICT EVALUATION CRITERIA:**
- Use "Yes" ONLY if you are 100% certain the rubric is FULLY satisfied
- Use "No" if there is ANY doubt, partial compliance, or uncertainty
- When uncertain, ALWAYS default to "No"
- A rubric is only "Yes" if it is completely and unambiguously met

**IMPORTANT: Your "ratings" list MUST contain exactly {num_rubrics} entries - one for each rubric item above, in the same order.**"""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_pass_flag(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if float(value) >= 0.5 else 0.0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "pass", "passed"}:
            return 1.0
        if lowered in {"false", "0", "no", "fail", "failed"}:
            return 0.0
    raise ValueError(f"Unsupported pass flag value: {value!r}")


def _truncate_pass_values(pass_values: list[float], num_rubrics: int) -> tuple[float, ...]:
    if num_rubrics <= 0:
        return tuple(pass_values)
    return tuple(pass_values[:num_rubrics])


def _extract_ratings_pass_values(payload: Mapping[str, Any]) -> list[float]:
    ratings = payload.get("ratings")
    if not isinstance(ratings, list):
        return []

    pass_values: list[float] = []
    for item in ratings:
        if isinstance(item, Mapping):
            if "rating" in item:
                try:
                    pass_values.append(_normalize_pass_flag(item["rating"]))
                except ValueError:
                    pass
                continue
            for pass_key in ("pass", "passed", "is_pass", "ok", "satisfied", "score"):
                if pass_key in item:
                    try:
                        pass_values.append(_normalize_pass_flag(item[pass_key]))
                    except ValueError:
                        pass
                    break
            continue
        if isinstance(item, (bool, int, float, str)):
            try:
                pass_values.append(_normalize_pass_flag(item))
            except ValueError:
                continue

    return pass_values


def _extract_pass_values(payload: Mapping[str, Any]) -> list[float]:
    ratings_pass_values = _extract_ratings_pass_values(payload)
    if ratings_pass_values:
        return ratings_pass_values

    pass_values: list[float] = []

    for key in ("per_rubric", "rubric_results", "rubric_passes", "results", "checks", "passes"):
        candidate = payload.get(key)
        if not isinstance(candidate, list):
            continue
        for item in candidate:
            if isinstance(item, (bool, int, float, str)):
                try:
                    pass_values.append(_normalize_pass_flag(item))
                except ValueError:
                    continue
                continue
            if isinstance(item, Mapping):
                for pass_key in ("pass", "passed", "is_pass", "ok", "satisfied", "score", "rating"):
                    if pass_key in item:
                        try:
                            pass_values.append(_normalize_pass_flag(item[pass_key]))
                        except ValueError:
                            pass
                        break
        if pass_values:
            break

    return pass_values


def _build_parse_result(
    pass_values: list[float], *, num_rubrics: int, parse_ok: bool, used_fallback: bool
) -> "JudgeParseResult":
    per_rubric_pass = _truncate_pass_values(pass_values, num_rubrics)
    num_rubrics_scored = len(per_rubric_pass)
    num_rubrics_expected = max(num_rubrics, 0)
    if num_rubrics_expected == 0:
        coverage_ratio = 1.0
    else:
        coverage_ratio = num_rubrics_scored / num_rubrics_expected

    return JudgeParseResult(
        pass_rate=_clamp01(_mean(per_rubric_pass) if per_rubric_pass else 0.0),
        per_rubric_pass=per_rubric_pass,
        parse_ok=parse_ok,
        used_fallback=used_fallback,
        num_rubrics_scored=num_rubrics_scored,
        num_rubrics_expected=num_rubrics_expected,
        coverage_ratio=coverage_ratio,
        coverage_ok=False,
    )


def _json_candidates(response_text: str) -> list[str]:
    candidates: list[str] = []
    stripped = response_text.strip()
    if stripped:
        candidates.append(stripped)

    for match in _JSON_BLOCK_RE.findall(response_text):
        candidate = match.strip()
        if candidate:
            candidates.append(candidate)

    first_brace = response_text.find("{")
    last_brace = response_text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = response_text[first_brace : last_brace + 1].strip()
        if candidate:
            candidates.append(candidate)

    deduped: list[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            deduped.append(candidate)
            seen.add(candidate)
    return deduped


@dataclass(frozen=True)
class JudgeParseResult:
    pass_rate: float
    per_rubric_pass: tuple[float, ...]
    parse_ok: bool
    used_fallback: bool
    num_rubrics_scored: int
    num_rubrics_expected: int
    coverage_ratio: float
    coverage_ok: bool


def _parse_json_payload(
    payload: Any, *, num_rubrics: int, used_fallback: bool
) -> JudgeParseResult:
    if isinstance(payload, list):
        pass_values: list[float] = []
        for item in payload:
            try:
                pass_values.append(_normalize_pass_flag(item))
            except ValueError:
                continue
        return _build_parse_result(
            pass_values,
            num_rubrics=num_rubrics,
            parse_ok=bool(pass_values),
            used_fallback=used_fallback,
        )

    if not isinstance(payload, Mapping):
        raise ValueError("JSON payload must be an object or list")

    pass_values = _extract_pass_values(payload)
    return _build_parse_result(
        pass_values,
        num_rubrics=num_rubrics,
        parse_ok=bool(pass_values),
        used_fallback=used_fallback,
    )


def _fallback_parse(response_text: str, *, num_rubrics: int) -> JudgeParseResult:
    pass_values: list[float] = []
    for match in _RATING_FIELD_RE.findall(response_text):
        pass_values.append(_normalize_pass_flag(match))
    if not pass_values:
        for match in _PASS_FIELD_RE.findall(response_text):
            pass_values.append(_normalize_pass_flag(match))

    parse_ok = bool(pass_values)
    return _build_parse_result(
        pass_values,
        num_rubrics=num_rubrics,
        parse_ok=parse_ok,
        used_fallback=True,
    )


def parse_judge_response(response_text: str, num_rubrics: int) -> JudgeParseResult:
    """
    Parse a grader response and produce a normalized pass-rate reward.

    We first try strict JSON parsing. If that fails, we fallback to regex extraction.
    """
    for idx, candidate in enumerate(_json_candidates(response_text)):
        try:
            payload = json.loads(candidate)
            return _parse_json_payload(payload, num_rubrics=num_rubrics, used_fallback=idx > 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return _fallback_parse(response_text, num_rubrics=num_rubrics)


def _apply_coverage_threshold(
    parse_result: JudgeParseResult, min_coverage_ratio: float
) -> JudgeParseResult:
    return replace(
        parse_result,
        coverage_ok=(parse_result.coverage_ratio >= min_coverage_ratio),
    )


def build_judge_prompt(
    *, instruction: str, candidate_response: str, rubrics: Sequence[str]
) -> list[renderers.Message]:
    rubric_lines = "\n".join(f"{i + 1}. {rubric}" for i, rubric in enumerate(rubrics))
    prompt = JUDGE_USER_TEMPLATE.format(
        prompt=instruction,
        completion=candidate_response,
        num_rubrics=len(rubrics),
        rubrics=rubric_lines,
    )
    return [{"role": "user", "content": prompt}]


def _is_openai_gpt5_model(model_name: str) -> bool:
    normalized = model_name.split(":")[0].strip().lower()
    if "/" in normalized:
        normalized = normalized.split("/")[-1]
    return normalized.startswith("gpt-5")


def _extract_openai_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    response_outputs = getattr(response, "output", None)
    text_parts: list[str] = []
    if isinstance(response_outputs, list):
        for output in response_outputs:
            content_items = getattr(output, "content", None)
            if content_items is None and isinstance(output, Mapping):
                content_items = output.get("content")
            if not isinstance(content_items, list):
                continue

            for content_item in content_items:
                text_value = getattr(content_item, "text", None)
                if text_value is None and isinstance(content_item, Mapping):
                    text_value = content_item.get("text")
                if isinstance(text_value, str) and text_value:
                    text_parts.append(text_value)

    extracted = "".join(text_parts).strip()
    if extracted:
        return extracted
    raise ValueError("OpenAI Responses API returned no text content.")


class OpenAIResponsesMessageCompleter(MessageCompleter):
    """Message completer backed by OpenAI Responses API."""

    def __init__(
        self,
        *,
        model_name: str,
        max_output_tokens: int,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for GPT-5 judge models. Install via `uv add openai`."
            ) from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set when using GPT-5 judge models."
            )

        self._client = AsyncOpenAI()
        self._model_name = model_name
        self._max_output_tokens = max_output_tokens

    async def __call__(self, messages: list[renderers.Message]) -> renderers.Message:
        if not messages:
            raise ValueError("OpenAI judge call requires at least one message.")

        if len(messages) == 1:
            input_text = renderers.format_content_as_string(messages[0]["content"])
        else:
            formatted_messages = []
            for message in messages:
                role = message.get("role", "user")
                content = renderers.format_content_as_string(message.get("content", ""))
                formatted_messages.append(f"{role}:\n{content}")
            input_text = "\n\n".join(formatted_messages)

        response = await self._client.responses.create(
            model=self._model_name,
            input=input_text,
            max_output_tokens=self._max_output_tokens,
        )
        content = _extract_openai_responses_text(response)
        return {"role": "assistant", "content": content}


def build_grader_llm(
    *,
    grader_llm_name: str,
    policy_renderer_name: str,
    policy_model_name_for_tokenizer: str,
    base_url: str | None = None,
    grader_renderer_name: str | None = None,
    grader_tokenizer_model_name: str | None = None,
    grader_max_tokens: int = 1024,
) -> MessageCompleter:
    if _is_openai_gpt5_model(grader_llm_name):
        if base_url is not None:
            warnings.warn(
                "Ignoring `base_url` for GPT-5 judge model. "
                "GPT-5 judge calls use OpenAI Responses API + OPENAI_API_KEY.",
                stacklevel=2,
            )
        return OpenAIResponsesMessageCompleter(
            model_name=grader_llm_name,
            max_output_tokens=grader_max_tokens,
        )

    resolved_renderer_name = grader_renderer_name
    if resolved_renderer_name is None:
        try:
            resolved_renderer_name = model_info.get_recommended_renderer_name(grader_llm_name)
        except Exception:
            resolved_renderer_name = policy_renderer_name
            warnings.warn(
                f"Could not infer a renderer for grader model `{grader_llm_name}`. "
                f"Falling back to policy renderer `{policy_renderer_name}`.",
                stacklevel=2,
            )

    tokenizer_model_name = grader_tokenizer_model_name or grader_llm_name
    try:
        tokenizer = get_tokenizer(tokenizer_model_name)
    except Exception:
        warnings.warn(
            f"Could not load tokenizer for `{tokenizer_model_name}`. "
            f"Falling back to policy tokenizer `{policy_model_name_for_tokenizer}`.",
            stacklevel=2,
        )
        tokenizer = get_tokenizer(policy_model_name_for_tokenizer)
        if grader_renderer_name is None:
            resolved_renderer_name = policy_renderer_name

    grader_renderer = get_renderer(name=resolved_renderer_name, tokenizer=tokenizer)
    service_client = tinker.ServiceClient(base_url=base_url)
    sampling_client = service_client.create_sampling_client(base_model=grader_llm_name)
    return TinkerMessageCompleter(
        sampling_client=sampling_client,
        renderer=grader_renderer,
        max_tokens=grader_max_tokens,
    )


class RLVRIfEnv(Env):
    def __init__(
        self,
        renderer: Renderer,
        datapoint: RLVRIfDatapoint,
        grader_llm: MessageCompleter,
        format_penalty_coef: float = 0.05,
        judge_max_retries: int = 3,
        judge_min_coverage_ratio: float = 0.85,
        judge_retry_delay_seconds: float = 0.0,
        debug: bool = False,
    ):
        self.renderer = renderer
        self.datapoint = datapoint
        self.grader_llm = grader_llm
        self.format_penalty_coef = format_penalty_coef
        self.judge_max_retries = max(1, judge_max_retries)
        self.judge_min_coverage_ratio = max(0.0, min(1.0, judge_min_coverage_ratio))
        self.judge_retry_delay_seconds = max(0.0, judge_retry_delay_seconds)
        self.debug = debug

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    def _policy_conversation(self) -> list[renderers.Message]:
        return list(self.datapoint.convo_prefix) + [{"role": "user", "content": self.datapoint.prompt}]

    async def initial_observation(self) -> tuple[ModelInput, StopCondition]:
        return self.renderer.build_generation_prompt(self._policy_conversation()), self.stop_condition

    def _failed_parse_result(self) -> JudgeParseResult:
        expected = len(self.datapoint.rubric_items)
        return JudgeParseResult(
            pass_rate=0.0,
            per_rubric_pass=(),
            parse_ok=False,
            used_fallback=False,
            num_rubrics_scored=0,
            num_rubrics_expected=expected,
            coverage_ratio=0.0 if expected > 0 else 1.0,
            coverage_ok=False,
        )

    async def _judge_response(
        self, policy_response: str
    ) -> tuple[JudgeParseResult, float, int, bool, str | None]:
        judge_prompt = build_judge_prompt(
            instruction=self.datapoint.prompt,
            candidate_response=policy_response,
            rubrics=self.datapoint.rubric_items,
        )
        start = time.perf_counter()
        last_parse_result: JudgeParseResult | None = None
        last_failure_reason: str | None = None

        for attempt in range(1, self.judge_max_retries + 1):
            try:
                grader_response = await self.grader_llm(judge_prompt)
                grader_content = renderers.get_text_content(grader_response)
                parse_result = parse_judge_response(
                    grader_content, num_rubrics=len(self.datapoint.rubric_items)
                )
                parse_result = _apply_coverage_threshold(
                    parse_result, min_coverage_ratio=self.judge_min_coverage_ratio
                )
                last_parse_result = parse_result

                if self.debug:
                    print("-" * 80)
                    print(f"Judge attempt {attempt}/{self.judge_max_retries}")
                    print("Judge prompt:")
                    print(judge_prompt[0]["content"])
                    print("\nJudge raw response:")
                    print(grader_content)
                    print("\nParsed:")
                    print(parse_result)

                if parse_result.parse_ok and parse_result.coverage_ok:
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    return parse_result, latency_ms, attempt, True, None

                if not parse_result.parse_ok:
                    last_failure_reason = "judge output missing per-rubric ratings"
                else:
                    last_failure_reason = (
                        f"judge rubric coverage {parse_result.coverage_ratio:.3f} "
                        f"is below threshold {self.judge_min_coverage_ratio:.3f}"
                    )
            except Exception as exc:
                last_failure_reason = f"{type(exc).__name__}: {exc}"

            if attempt < self.judge_max_retries and self.judge_retry_delay_seconds > 0:
                await asyncio.sleep(self.judge_retry_delay_seconds)

        if last_parse_result is None:
            last_parse_result = self._failed_parse_result()
        else:
            last_parse_result = _apply_coverage_threshold(
                last_parse_result, min_coverage_ratio=self.judge_min_coverage_ratio
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        return (
            last_parse_result,
            latency_ms,
            self.judge_max_retries,
            False,
            last_failure_reason,
        )

    async def step(self, action: Action) -> StepResult:
        policy_action_message, parse_success = self.renderer.parse_response(action)
        policy_response = renderers.get_text_content(policy_action_message)
        (
            parse_result,
            judge_latency_ms,
            judge_attempt_count,
            judge_call_success,
            judge_failure_reason,
        ) = await self._judge_response(policy_response)

        format_ok = float(parse_success)
        rollout_valid = float(judge_call_success)
        if judge_call_success:
            reward = parse_result.pass_rate + self.format_penalty_coef * (format_ok - 1.0)
        else:
            # Invalid rollouts are filtered before advantages/training data assembly.
            reward = 0.0

        convo = self._policy_conversation() + [policy_action_message]
        metrics: dict[str, float | int] = {
            "format_ok": format_ok,
            "rollout_valid": rollout_valid,
            "judge_call_success": rollout_valid,
            "pass_rate": parse_result.pass_rate,
            "num_rubrics": len(self.datapoint.rubric_items),
            "pass_count": int(round(sum(parse_result.per_rubric_pass))),
            "judge_latency_ms": judge_latency_ms,
            "judge_attempt_count": judge_attempt_count,
            "judge_retry_count": judge_attempt_count - 1,
            "judge_parse_ok": float(parse_result.parse_ok),
            "judge_used_fallback": float(parse_result.used_fallback),
            "judge_num_rubrics_scored": parse_result.num_rubrics_scored,
            "judge_coverage_ratio": parse_result.coverage_ratio,
            "judge_coverage_ok": float(parse_result.coverage_ok),
        }

        logs: dict[str, str | int | float] = {}
        if self.datapoint.sample_id is not None:
            logs["sample_id"] = self.datapoint.sample_id
        if judge_failure_reason is not None:
            logs["judge_failure_reason"] = judge_failure_reason

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=self.renderer.build_generation_prompt(convo),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
            logs=logs,
        )


@dataclass(frozen=True)
class RLVRIfEnvGroupBuilder(EnvGroupBuilder):
    renderer: Renderer
    datapoint: RLVRIfDatapoint
    grader_llm: MessageCompleter
    group_size: int
    format_penalty_coef: float
    judge_max_retries: int
    judge_min_coverage_ratio: float
    judge_retry_delay_seconds: float

    async def make_envs(self) -> Sequence[RLVRIfEnv]:
        return [
            RLVRIfEnv(
                renderer=self.renderer,
                datapoint=self.datapoint,
                grader_llm=self.grader_llm,
                format_penalty_coef=self.format_penalty_coef,
                judge_max_retries=self.judge_max_retries,
                judge_min_coverage_ratio=self.judge_min_coverage_ratio,
                judge_retry_delay_seconds=self.judge_retry_delay_seconds,
            )
            for _ in range(self.group_size)
        ]


@dataclass(frozen=True)
class RLVRIfDataset(RLDataset):
    renderer: Renderer
    batch_size: int
    group_size: int
    datapoints: Sequence[RLVRIfDatapoint]
    grader_llm: MessageCompleter
    format_penalty_coef: float = 0.05
    judge_max_retries: int = 3
    judge_min_coverage_ratio: float = 0.85
    judge_retry_delay_seconds: float = 0.0

    def get_batch(self, index: int) -> Sequence[RLVRIfEnvGroupBuilder]:
        return [
            RLVRIfEnvGroupBuilder(
                renderer=self.renderer,
                datapoint=self.datapoints[index * self.batch_size + i],
                grader_llm=self.grader_llm,
                group_size=self.group_size,
                format_penalty_coef=self.format_penalty_coef,
                judge_max_retries=self.judge_max_retries,
                judge_min_coverage_ratio=self.judge_min_coverage_ratio,
                judge_retry_delay_seconds=self.judge_retry_delay_seconds,
            )
            for i in range(self.batch_size)
        ]

    def __len__(self) -> int:
        return len(self.datapoints) // self.batch_size


@chz.chz
class RLVRIfDatasetBuilder(RLDatasetBuilder):
    renderer_name: str
    model_name_for_tokenizer: str
    batch_size: int
    train_group_size: int
    test_group_size: int = 1

    train_datapoint_list_builder: RLVRIfDatapointListBuilder
    test_datapoint_list_builder: RLVRIfDatapointListBuilder | None = None

    grader_llm_name: str = "gpt-5-mini"
    grader_renderer_name: str | None = None
    grader_tokenizer_model_name: str | None = None
    grader_max_tokens: int = 1024

    base_url: str | None = None
    format_penalty_coef: float = 0.05
    judge_max_retries: int = 3
    judge_min_coverage_ratio: float = 0.85
    judge_retry_delay_seconds: float = 0.0
    seed: int = 0
    shuffle_train_data: bool = True

    def _get_grader_llm(self) -> MessageCompleter:
        return build_grader_llm(
            grader_llm_name=self.grader_llm_name,
            policy_renderer_name=self.renderer_name,
            policy_model_name_for_tokenizer=self.model_name_for_tokenizer,
            base_url=self.base_url,
            grader_renderer_name=self.grader_renderer_name,
            grader_tokenizer_model_name=self.grader_tokenizer_model_name,
            grader_max_tokens=self.grader_max_tokens,
        )

    async def __call__(self) -> tuple[RLVRIfDataset, RLVRIfDataset | None]:
        train_datapoints = list(self.train_datapoint_list_builder())
        if self.shuffle_train_data:
            random.Random(self.seed).shuffle(train_datapoints)
        if len(train_datapoints) < self.batch_size:
            raise ValueError(
                f"Train dataset has {len(train_datapoints)} datapoints but batch_size is {self.batch_size}. "
                "Increase data size or decrease `groups_per_batch`."
            )

        test_datapoints = (
            list(self.test_datapoint_list_builder())
            if self.test_datapoint_list_builder is not None
            else None
        )

        renderer = get_renderer(
            name=self.renderer_name, tokenizer=get_tokenizer(self.model_name_for_tokenizer)
        )
        grader_llm = self._get_grader_llm()

        train_dataset = RLVRIfDataset(
            renderer=renderer,
            batch_size=self.batch_size,
            group_size=self.train_group_size,
            datapoints=train_datapoints,
            grader_llm=grader_llm,
            format_penalty_coef=self.format_penalty_coef,
            judge_max_retries=self.judge_max_retries,
            judge_min_coverage_ratio=self.judge_min_coverage_ratio,
            judge_retry_delay_seconds=self.judge_retry_delay_seconds,
        )

        if test_datapoints is None:
            return train_dataset, None
        if not test_datapoints:
            raise ValueError("Test datapoint builder returned no datapoints")

        test_dataset = RLVRIfDataset(
            renderer=renderer,
            batch_size=len(test_datapoints),
            group_size=self.test_group_size,
            datapoints=test_datapoints,
            grader_llm=grader_llm,
            format_penalty_coef=self.format_penalty_coef,
            judge_max_retries=self.judge_max_retries,
            judge_min_coverage_ratio=self.judge_min_coverage_ratio,
            judge_retry_delay_seconds=self.judge_retry_delay_seconds,
        )
        return train_dataset, test_dataset
