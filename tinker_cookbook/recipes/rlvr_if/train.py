import asyncio
from datetime import datetime
from typing import Any

import chz
from tinker.types import LossFnType
from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.recipes.rlvr_if.data import RLVRIfDatapointListBuilderFromJson
from tinker_cookbook.recipes.rlvr_if.env import RLVRIfDatasetBuilder
from tinker_cookbook.rl.train import AsyncConfig, Config, main
from tinker_cookbook.rl.types import RLDatasetBuilder


@chz.chz
class CLIConfig:
    # Model configuration
    model_name: str = "moonshotai/Kimi-K2-Thinking"
    lora_rank: int = 32
    renderer_name: str | None = None
    load_checkpoint_path: str | None = None

    seed: int = 0

    # Training hyperparameters
    train_group_size: int = 16
    test_group_size: int = 16
    groups_per_batch: int = 32
    batch_size: int = 32
    group_size: int = 16
    learning_rate: float = 1e-5
    max_tokens: int = 8192
    temperature: float = 1.0
    kl_penalty_coef: float = 0.0
    num_substeps: int = 1

    # Judge configuration
    grader_llm_name: str = "gpt-5-mini"
    grader_renderer_name: str | None = None
    grader_tokenizer_model_name: str | None = None
    grader_max_tokens: int = 4096
    judge_max_retries: int = 3
    judge_min_coverage_ratio: float = 0.85
    judge_retry_delay_seconds: float = 0.0
    format_penalty_coef: float = 0.05

    # Dataset paths
    train_data_path: str = "tinker_cookbook/recipes/rlvr_if/IF_1K_filtered_train.jsonl"
    test_data_path: str | None = "tinker_cookbook/recipes/rlvr_if/IF_1K_filtered_test.jsonl"
    train_split: str | None = None
    test_split: str | None = None
    max_train_datapoints: int | None = None
    max_test_datapoints: int | None = None
    shuffle_train_data: bool = True

    # Logging and eval
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None
    compute_post_kl: bool = False
    eval_every: int = 10
    save_every: int = 10
    behavior_if_log_dir_exists: cli_utils.LogdirBehavior = "ask"

    # Service and async
    base_url: str | None = None
    max_steps_off_policy: int | None = None

    # Loss function
    loss_fn: LossFnType = "importance_sampling"
    loss_fn_config: dict[str, Any] | None = None


def get_dataset_builder(
    *,
    batch_size: int,
    policy_model_name: str,
    renderer_name: str,
    train_group_size: int,
    test_group_size: int,
    train_data_path: str,
    test_data_path: str | None,
    train_split: str | None,
    test_split: str | None,
    max_train_datapoints: int | None,
    max_test_datapoints: int | None,
    grader_llm_name: str,
    grader_renderer_name: str | None,
    grader_tokenizer_model_name: str | None,
    grader_max_tokens: int,
    judge_max_retries: int,
    judge_min_coverage_ratio: float,
    judge_retry_delay_seconds: float,
    format_penalty_coef: float,
    base_url: str | None,
    seed: int,
    shuffle_train_data: bool,
) -> RLDatasetBuilder:
    train_builder = RLVRIfDatapointListBuilderFromJson(
        data_path=train_data_path,
        split=train_split,
        max_datapoints=max_train_datapoints,
    )

    effective_test_data_path = test_data_path
    if effective_test_data_path is None and test_split is not None:
        effective_test_data_path = train_data_path

    test_builder = (
        RLVRIfDatapointListBuilderFromJson(
            data_path=effective_test_data_path,
            split=test_split,
            max_datapoints=max_test_datapoints,
        )
        if effective_test_data_path is not None
        else None
    )

    return RLVRIfDatasetBuilder(
        batch_size=batch_size,
        model_name_for_tokenizer=policy_model_name,
        renderer_name=renderer_name,
        train_group_size=train_group_size,
        test_group_size=test_group_size,
        train_datapoint_list_builder=train_builder,
        test_datapoint_list_builder=test_builder,
        grader_llm_name=grader_llm_name,
        grader_renderer_name=grader_renderer_name,
        grader_tokenizer_model_name=grader_tokenizer_model_name,
        grader_max_tokens=grader_max_tokens,
        judge_max_retries=judge_max_retries,
        judge_min_coverage_ratio=judge_min_coverage_ratio,
        judge_retry_delay_seconds=judge_retry_delay_seconds,
        format_penalty_coef=format_penalty_coef,
        base_url=base_url,
        seed=seed,
        shuffle_train_data=shuffle_train_data,
    )


async def cli_main(cli_config: CLIConfig):
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=cli_config.model_name,
        explicit_renderer_name=cli_config.renderer_name,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        base_url=cli_config.base_url,
    )

    model_name_short = cli_config.model_name.replace("/", "-")
    run_name = (
        f"rlvr_if-{model_name_short}-lr{cli_config.learning_rate}-"
        f"gs{cli_config.train_group_size}-bs{cli_config.groups_per_batch}-"
        f"judge{cli_config.grader_llm_name.replace('/', '-')}-seed{cli_config.seed}-"
        f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    )

    log_path = cli_config.log_path or f"/tmp/tinker-examples/rlvr_if/{run_name}"
    wandb_name = cli_config.wandb_name or run_name

    config = Config(
        learning_rate=cli_config.learning_rate,
        dataset_builder=get_dataset_builder(
            batch_size=cli_config.groups_per_batch,
            policy_model_name=cli_config.model_name,
            renderer_name=renderer_name,
            train_group_size=cli_config.train_group_size,
            test_group_size=cli_config.test_group_size,
            train_data_path=cli_config.train_data_path,
            test_data_path=cli_config.test_data_path,
            train_split=cli_config.train_split,
            test_split=cli_config.test_split,
            max_train_datapoints=cli_config.max_train_datapoints,
            max_test_datapoints=cli_config.max_test_datapoints,
            grader_llm_name=cli_config.grader_llm_name,
            grader_renderer_name=cli_config.grader_renderer_name,
            grader_tokenizer_model_name=cli_config.grader_tokenizer_model_name,
            grader_max_tokens=cli_config.grader_max_tokens,
            judge_max_retries=cli_config.judge_max_retries,
            judge_min_coverage_ratio=cli_config.judge_min_coverage_ratio,
            judge_retry_delay_seconds=cli_config.judge_retry_delay_seconds,
            format_penalty_coef=cli_config.format_penalty_coef,
            base_url=cli_config.base_url,
            seed=cli_config.seed,
            shuffle_train_data=cli_config.shuffle_train_data,
        ),
        model_name=cli_config.model_name,
        renderer_name=renderer_name,
        lora_rank=cli_config.lora_rank,
        max_tokens=cli_config.max_tokens,
        temperature=cli_config.temperature,
        wandb_project=cli_config.wandb_project,
        wandb_name=wandb_name,
        log_path=log_path,
        base_url=cli_config.base_url,
        load_checkpoint_path=cli_config.load_checkpoint_path,
        compute_post_kl=cli_config.compute_post_kl,
        kl_penalty_coef=cli_config.kl_penalty_coef,
        num_substeps=cli_config.num_substeps,
        eval_every=cli_config.eval_every,
        save_every=cli_config.save_every,
        async_config=AsyncConfig(
            max_steps_off_policy=cli_config.max_steps_off_policy,
            groups_per_batch=cli_config.groups_per_batch,
        )
        if cli_config.max_steps_off_policy is not None
        else None,
        loss_fn=cli_config.loss_fn,
        loss_fn_config=cli_config.loss_fn_config,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists=cli_config.behavior_if_log_dir_exists)
    await main(config)


if __name__ == "__main__":
    cli_config = chz.entrypoint(CLIConfig)
    asyncio.run(cli_main(cli_config))
