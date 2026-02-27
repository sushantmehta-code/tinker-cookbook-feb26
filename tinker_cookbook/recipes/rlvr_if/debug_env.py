import asyncio

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.rlvr_if.data import RLVRIfDatapointListBuilderFromJson
from tinker_cookbook.recipes.rlvr_if.env import RLVRIfEnv, build_grader_llm
from tinker_cookbook.rl.rollouts import do_single_rollout
from tinker_cookbook.tokenizer_utils import get_tokenizer


async def main(
    *,
    data_path: str = "tinker_cookbook/recipes/rlvr_if/IF_1K_filtered_train.jsonl",
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507",
    grader_llm_name: str = "gpt-5-mini",
    base_url: str | None = None,
):
    datapoint = RLVRIfDatapointListBuilderFromJson(data_path=data_path, max_datapoints=1)()[0]

    service_client = tinker.ServiceClient(base_url=base_url)
    policy = TinkerTokenCompleter(
        sampling_client=service_client.create_sampling_client(base_model=model_name),
        max_tokens=512,
    )

    policy_renderer_name = model_info.get_recommended_renderer_name(model_name)
    policy_renderer = renderers.get_renderer(
        policy_renderer_name, tokenizer=get_tokenizer(model_name)
    )
    grader = build_grader_llm(
        grader_llm_name=grader_llm_name,
        policy_renderer_name=policy_renderer_name,
        policy_model_name_for_tokenizer=model_name,
        base_url=base_url,
    )

    env = RLVRIfEnv(
        renderer=policy_renderer,
        datapoint=datapoint,
        grader_llm=grader,
        format_penalty_coef=0.05,
        debug=True,
    )
    await do_single_rollout(policy, env)


if __name__ == "__main__":
    asyncio.run(main())
