import pytest

from run_experiments import (
    run_experiments,
    summarize_results,
)


def create_result(
    reward: float,
    autonomy: float,
) -> dict:
    """Crée un résultat expérimental minimal complet."""

    return {
        "policy": "dqn",
        "reward": reward,
        "grid_import_kwh": 100.0,
        "unmet_demand_kwh": 5.0,
        "battery_throughput_kwh": 20.0,
        "battery_degradation_cost": 0.2,
        "equivalent_battery_cycles": 1.0,
        "autonomy_rate_percent": autonomy,
        "demand_satisfaction_percent": 99.0,
        "seed": 42,
    }


def test_summary_computes_mean_and_standard_deviation():
    results = [
        create_result(
            reward=10.0,
            autonomy=50.0,
        ),
        create_result(
            reward=14.0,
            autonomy=60.0,
        ),
    ]

    summary = summarize_results(
        results
    )

    assert len(summary) == 1

    row = summary.iloc[0]

    assert row["policy"] == "dqn"
    assert row["reward_mean"] == pytest.approx(
        12.0
    )
    assert row["reward_std"] == pytest.approx(
        2.828427
    )
    assert row[
        "autonomy_rate_percent_mean"
    ] == pytest.approx(
        55.0
    )
    assert row[
        "autonomy_rate_percent_std"
    ] == pytest.approx(
        7.071068
    )


def test_empty_results_are_rejected():
    with pytest.raises(ValueError):
        summarize_results([])


def test_missing_summary_metrics_are_rejected():
    incomplete_results = [
        {
            "policy": "dqn",
            "reward": 10.0,
        }
    ]

    with pytest.raises(ValueError):
        summarize_results(
            incomplete_results
        )


def test_experiments_require_at_least_one_seed():
    with pytest.raises(ValueError):
        run_experiments(
            seeds=[],
            episodes=1,
        )


def test_experiment_seeds_must_be_unique():
    with pytest.raises(ValueError):
        run_experiments(
            seeds=[
                42,
                42,
            ],
            episodes=1,
        )


def test_episode_count_must_be_positive():
    with pytest.raises(ValueError):
        run_experiments(
            seeds=[
                42,
            ],
            episodes=0,
        )