# RLVR IF: RL with Verifiable Rubrics for Instruction Following

This recipe trains instruction-following models with RL using rubric-based rewards.
For each rollout:

1. The policy model generates a response to a prompt.
2. A judge model grades that response against all rubric items in **one call**.
3. Reward is the rubric pass rate (0-1), with an optional small format penalty.

## Files

- `data.py`: datapoint schema + robust JSON/JSONL dataset loader.
- `env.py`: RL environment, single-call judge prompt, response parser, dataset builder.
- `train.py`: CLI entrypoint for end-to-end RL training.
- `debug_env.py`: one-sample debug run that prints judge prompt/response/parser output.

## Dataset Format

The loader accepts either a top-level list, JSONL file, or a split dict (`train`/`test`).

Each datapoint should include:

- `prompt` (or `instruction` / `question`)
- `rubrics` (or `rubric_items`) as a list of strings or rubric objects

For wide tabular JSONL formats, the loader also supports rubric columns named like
`rubric - 1. criterion`, `rubric - 2. criterion`, etc.

Minimal JSON example:

```json
[
  {
    "id": "sample-1",
    "prompt": "Write a short reply recommending one healthy breakfast.",
    "rubrics": [
      "Mentions exactly one breakfast option",
      "Includes at least one protein-rich ingredient",
      "Under 60 words"
    ]
  }
]
```

Rubric object form is also supported:

```json
{
  "prompt": "Summarize this budget issue.",
  "rubric_items": [
    {"rubric_str": "Names the responsible council types"},
    {"text": "Identifies the four exceptions to the standard rule"}
  ]
}
```

## Train

```bash
python -m tinker_cookbook.recipes.rlvr_if.train \
  model_name="Qwen/Qwen3-4B-Instruct-2507" \
  train_data_path="tinker_cookbook/recipes/rlvr_if/IF_1K_filtered_train.jsonl" \
  test_data_path="tinker_cookbook/recipes/rlvr_if/IF_1K_filtered_test.jsonl" \
  grader_llm_name="gpt-5-mini" \
  groups_per_batch=64 train_group_size=4 learning_rate=1e-5
```

Optional test split from the same file:

```bash
python -m tinker_cookbook.recipes.rlvr_if.train \
  train_data_path="tinker_cookbook/recipes/rlvr_if/IF_1K_filtered.jsonl" \
  train_split="train" test_split="test"
```

## Debug One Sample

```bash
python -m tinker_cookbook.recipes.rlvr_if.debug_env
```

This prints:

- judge prompt content,
- raw judge response,
- parsed rubric pass results and pass rate.

## Judge Output Format

The judge prompt requests JSON with a `ratings` list containing exactly one entry per rubric,
in rubric order:

```json
{
  "ratings": [
    {"rating": "Yes", "rationale": "brief reason"},
    {"rating": "No", "rationale": "brief reason"}
  ]
}
```

For each item, `rating` should be exactly `"Yes"` or `"No"`. The parser is optimized for this
format and keeps backward compatibility with legacy `per_rubric`/`pass` outputs.

## GPT-5 Judge Backend

When `grader_llm_name` is a GPT-5 model (for example `gpt-5-mini`), RLVR IF sends judge calls
through the OpenAI Responses API instead of the Tinker sampling client.

Required runtime setup:

- `OPENAI_API_KEY` must be set in the environment.
- The `openai` Python package must be available in the runtime.

This avoids tokenizer fallback warnings and the `Sampling is not supported for gpt-5-mini` error.

## Edge-case Handling

- Judge calls retry up to `judge_max_retries` (default `3`).
- A judge result is accepted only when rubric coverage is at least `judge_min_coverage_ratio` (default `0.85`).
- Reward uses only rubric ratings that were actually returned by the judge (no default scores for missing rubric ratings).
- If judge retries are exhausted, that rollout is marked invalid (`rollout_valid=0`) and excluded from sync-training advantage/data assembly.
- If an entire group (or full batch) has no valid rollouts, training skips that group/batch and continues instead of crashing the job.
