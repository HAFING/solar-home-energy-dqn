import pandas as pd
import pytest

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


def test_reset_returns_state(sample_data):
    env = EnergyEnvironment(sample_data)

    state = env.reset()

    assert len(state) == 6


def test_initial_soc(sample_data):
    env = EnergyEnvironment(sample_data)

    env.reset()

    assert env.soc == 0.5


def test_charge_increases_soc():
    df = pd.DataFrame(
        {
            "pv_production": [5],
            "consumption": [2],
            "grid_available": [1],
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    old_soc = env.soc

    env.step(env.CHARGE)

    assert env.soc > old_soc


def test_discharge_decreases_soc():
    df = pd.DataFrame(
        {
            "pv_production": [1],
            "consumption": [4],
            "grid_available": [1],
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    old_soc = env.soc

    env.step(env.DISCHARGE)

    assert env.soc < old_soc


def test_soc_never_above_one():
    df = pd.DataFrame(
        {
            "pv_production": [20] * 20,
            "consumption": [0] * 20,
            "grid_available": [1] * 20,
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    for _ in range(len(df)):
        _, _, done, _ = env.step(
            env.CHARGE
        )

        if done:
            break

    assert env.soc <= 1.0


def test_soc_never_below_zero():
    df = pd.DataFrame(
        {
            "pv_production": [0] * 20,
            "consumption": [20] * 20,
            "grid_available": [1] * 20,
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    for _ in range(len(df)):
        _, _, done, _ = env.step(
            env.DISCHARGE
        )

        if done:
            break

    assert env.soc >= 0.0


def test_invalid_action(sample_data):
    env = EnergyEnvironment(sample_data)

    env.reset()

    with pytest.raises(ValueError):
        env.step(99)


def test_grid_outage_creates_unmet_demand():

    df = pd.DataFrame(
        {
            "pv_production": [0],
            "consumption": [10],
            "grid_available": [0],
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    _, _, _, info = env.step(
        env.IDLE
    )

    assert info["unmet_demand"] > 0


def test_grid_available_imports_energy():

    df = pd.DataFrame(
        {
            "pv_production": [0],
            "consumption": [10],
            "grid_available": [1],
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    _, _, _, info = env.step(
        env.IDLE
    )

    assert info["grid_import"] > 0


def test_done_flag():

    df = pd.DataFrame(
        {
            "pv_production": [5],
            "consumption": [2],
            "grid_available": [1],
        }
    )

    env = EnergyEnvironment(df)

    env.reset()

    _, _, done, _ = env.step(
        env.IDLE
    )

    assert done is True