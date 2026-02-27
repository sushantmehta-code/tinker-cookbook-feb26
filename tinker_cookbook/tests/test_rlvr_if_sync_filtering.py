from types import SimpleNamespace

from tinker_cookbook.rl.train import (
    _filter_invalid_rollouts_from_group_pairs,
    _remove_constant_reward_group_pairs,
)
from tinker_cookbook.rl.types import TrajectoryGroup


def _make_trajectory(*, valid: bool, reward: float):
    transition = SimpleNamespace(metrics={"rollout_valid": 1.0 if valid else 0.0}, reward=reward)
    return SimpleNamespace(transitions=[transition], final_ob=SimpleNamespace(length=0))


def _make_group(valid_flags, rewards):
    trajectories = [
        _make_trajectory(valid=valid_flag, reward=reward)
        for valid_flag, reward in zip(valid_flags, rewards, strict=True)
    ]
    return TrajectoryGroup(
        trajectories_G=trajectories,
        final_rewards_G=[0.0 for _ in trajectories],
        metrics_G=[{} for _ in trajectories],
    )


def test_filter_invalid_rollouts_drops_all_invalid_groups():
    builders = [SimpleNamespace(name="g1"), SimpleNamespace(name="g2")]
    groups = [
        _make_group([True, False], [0.7, 0.2]),
        _make_group([False], [0.1]),
    ]

    filtered_builders, filtered_groups, stats = _filter_invalid_rollouts_from_group_pairs(
        builders, groups
    )

    assert len(filtered_builders) == 1
    assert len(filtered_groups) == 1
    assert filtered_builders[0].name == "g1"
    assert len(filtered_groups[0].trajectories_G) == 1
    assert stats["groups_dropped_all_invalid"] == 1
    assert stats["trajectories_dropped_invalid"] == 2


def test_remove_constant_reward_group_pairs_keeps_non_constant():
    builders = [SimpleNamespace(name="const"), SimpleNamespace(name="mixed")]
    groups = [
        _make_group([True, True], [0.5, 0.5]),
        _make_group([True, True], [0.1, 0.9]),
    ]

    filtered_builders, filtered_groups, dropped_constant_groups = _remove_constant_reward_group_pairs(
        builders, groups
    )

    assert dropped_constant_groups == 1
    assert len(filtered_builders) == 1
    assert len(filtered_groups) == 1
    assert filtered_builders[0].name == "mixed"
