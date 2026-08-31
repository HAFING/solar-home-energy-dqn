import gymnasium as gym
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment.energy_environment import (
    EnergyEnvironment,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "pv_production": [5, 1, 0, 6],
            "consumption": [2, 3, 4, 1],
            "grid_available": [1, 1, 0, 1],
        }
    )


def test_environment_inherits_from_gymnasium(sample_data):
    environment = EnergyEnvironment(sample_data)

    assert isinstance(environment, gym.Env)


def test_environment_follows_gymnasium_api(sample_data):
    environment = EnergyEnvironment(sample_data)

    check_env(
        environment,
        skip_render_check=True,
    )


def test_action_space(sample_data):
    environment = EnergyEnvironment(sample_data)

    assert isinstance(
        environment.action_space,
        gym.spaces.Discrete,
    )
    assert environment.action_space.n == 3


def test_observation_space(sample_data):
    environment = EnergyEnvironment(sample_data)

    observation, _ = environment.reset()

    assert isinstance(
        environment.observation_space,
        gym.spaces.Box,
    )
    assert environment.observation_space.shape == (6,)
    assert environment.observation_space.contains(observation)


def test_reset_returns_observation_and_info(sample_data):
    environment = EnergyEnvironment(sample_data)

    observation, info = environment.reset()

    assert len(observation) == 6
    assert isinstance(info, dict)


def test_initial_soc(sample_data):
    environment = EnergyEnvironment(sample_data)

    environment.reset()

    assert environment.soc == 0.5


def test_custom_initial_soc(sample_data):
    environment = EnergyEnvironment(
        sample_data,
        initial_soc=0.75,
    )

    observation, _ = environment.reset()

    assert environment.soc == 0.75
    assert observation[2] == pytest.approx(0.75)


def test_reset_option_changes_initial_soc(sample_data):
    environment = EnergyEnvironment(sample_data)

    observation, _ = environment.reset(
        options={
            "initial_soc": 0.25,
        }
    )

    assert environment.soc == 0.25
    assert observation[2] == pytest.approx(0.25)


def test_invalid_initial_soc(sample_data):
    with pytest.raises(ValueError):
        EnergyEnvironment(
            sample_data,
            initial_soc=1.5,
        )


def test_charge_increases_soc():
    data = pd.DataFrame(
        {
            "pv_production": [5],
            "consumption": [2],
            "grid_available": [1],
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    old_soc = environment.soc

    environment.step(environment.CHARGE)

    assert environment.soc > old_soc


def test_discharge_decreases_soc():
    data = pd.DataFrame(
        {
            "pv_production": [1],
            "consumption": [4],
            "grid_available": [1],
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    old_soc = environment.soc

    environment.step(environment.DISCHARGE)

    assert environment.soc < old_soc


def test_soc_never_above_one():
    data = pd.DataFrame(
        {
            "pv_production": [20] * 20,
            "consumption": [0] * 20,
            "grid_available": [1] * 20,
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    for _ in range(len(data)):
        (
            _,
            _,
            terminated,
            truncated,
            _,
        ) = environment.step(
            environment.CHARGE
        )

        if terminated or truncated:
            break

    assert environment.soc <= 1.0


def test_soc_never_below_zero():
    data = pd.DataFrame(
        {
            "pv_production": [0] * 20,
            "consumption": [20] * 20,
            "grid_available": [1] * 20,
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    for _ in range(len(data)):
        (
            _,
            _,
            terminated,
            truncated,
            _,
        ) = environment.step(
            environment.DISCHARGE
        )

        if terminated or truncated:
            break

    assert environment.soc >= 0.0


def test_invalid_action(sample_data):
    environment = EnergyEnvironment(sample_data)
    environment.reset()

    with pytest.raises(ValueError):
        environment.step(99)


def test_grid_outage_creates_unmet_demand():
    data = pd.DataFrame(
        {
            "pv_production": [0],
            "consumption": [10],
            "grid_available": [0],
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    _, _, _, _, info = environment.step(
        environment.IDLE
    )

    assert info["unmet_demand"] > 0


def test_grid_available_imports_energy():
    data = pd.DataFrame(
        {
            "pv_production": [0],
            "consumption": [10],
            "grid_available": [1],
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    _, _, _, _, info = environment.step(
        environment.IDLE
    )

    assert info["grid_import"] > 0


def test_termination_flag():
    data = pd.DataFrame(
        {
            "pv_production": [5],
            "consumption": [2],
            "grid_available": [1],
        }
    )

    environment = EnergyEnvironment(data)
    environment.reset()

    (
        _,
        _,
        terminated,
        truncated,
        _,
    ) = environment.step(
        environment.IDLE
    )

    assert terminated is True
    assert truncated is False