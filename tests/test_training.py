import numpy as np
import pandas as pd

from src.agent import DQNAgent
from src.environment.energy_environment import EnergyEnvironment
from train import evaluate_agent, run_training_episode


def create_small_environment(number_of_hours=48):
    data = pd.DataFrame(
        {
            "pv_production": np.tile(
                [
                    0.0,
                    0.0,
                    0.5,
                    1.5,
                    3.0,
                    4.0,
                    3.0,
                    1.0,
                ],
                number_of_hours // 8,
            ),
            "consumption": np.full(
                number_of_hours,
                0.8,
            ),
            "grid_available": np.ones(
                number_of_hours,
                dtype=int,
            ),
        }
    )

    return EnergyEnvironment(data)


def test_training_episode_returns_expected_metrics():
    environment = create_small_environment()

    agent = DQNAgent(
        batch_size=8,
        buffer_capacity=100,
        epsilon_decay=0.99,
        target_update_freq=5,
        device="cpu",
    )

    results = run_training_episode(environment, agent)

    assert set(results) == {
        "reward",
        "average_loss",
        "grid_import",
        "unmet_demand",
        "battery_discharge",
        "battery_throughput",
        "battery_degradation_cost",
        "final_soc",
    }

    assert len(agent.memory) == 48
    assert results["average_loss"] >= 0
    assert results["battery_throughput"] >= 0
    assert results["battery_degradation_cost"] >= 0
    assert 0 <= results["final_soc"] <= 1


def test_evaluation_does_not_add_experience():
    environment = create_small_environment()

    agent = DQNAgent(
        batch_size=8,
        device="cpu",
    )

    initial_memory_size = len(agent.memory)

    results = evaluate_agent(environment, agent)

    assert len(agent.memory) == initial_memory_size
    assert "reward" in results
    assert "grid_import" in results
    assert "unmet_demand" in results