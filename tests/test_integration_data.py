import numpy as np
import pandas as pd
import pytest

from src.integration_data import (
    compute_normalization_scales,
    load_environment_data,
    simulate_grid_availability,
    split_environment_data,
)


def test_simulated_grid_contains_available_and_outage_hours():
    grid = simulate_grid_availability(
        number_of_hours=1000,
        outage_start_probability=0.05,
        seed=42,
    )

    assert len(grid) == 1000
    assert set(np.unique(grid)).issubset({0, 1})
    assert 0 in grid
    assert 1 in grid


def test_grid_simulation_is_reproducible():
    first_grid = simulate_grid_availability(
        number_of_hours=500,
        seed=42,
    )

    second_grid = simulate_grid_availability(
        number_of_hours=500,
        seed=42,
    )

    assert np.array_equal(first_grid, second_grid)


def test_load_environment_data_creates_expected_columns(
    tmp_path,
):
    csv_path = tmp_path / "processed_energy.csv"

    sample = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-01",
                periods=48,
                freq="h",
            ),
            "load_kwh": np.full(48, 0.5),
            "pv_kwh": np.linspace(0, 5, 48),
        }
    )

    sample.to_csv(csv_path, index=False)

    result = load_environment_data(
        csv_path=csv_path,
        outage_start_probability=0.10,
        seed=42,
    )

    assert list(result.columns) == [
        "timestamp",
        "consumption",
        "pv_production",
        "grid_available",
    ]

    assert len(result) == 48
    assert result["consumption"].min() >= 0
    assert result["pv_production"].min() >= 0
    assert set(result["grid_available"].unique()).issubset(
        {0, 1}
    )


def test_split_environment_data_uses_full_days():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2023-01-01",
                periods=8760,
                freq="h",
            ),
            "consumption": np.ones(8760),
            "pv_production": np.ones(8760),
            "grid_available": np.ones(
                8760,
                dtype=int,
            ),
        }
    )

    train, validation, test = split_environment_data(data)

    assert len(train) == 255 * 24
    assert len(validation) == 55 * 24
    assert len(test) == 55 * 24

    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()


def test_invalid_outage_probability_is_rejected():
    with pytest.raises(ValueError):
        simulate_grid_availability(
            number_of_hours=100,
            outage_start_probability=1.5,
        )

def test_normalization_scales_use_training_data():
    training_data = pd.DataFrame(
        {
            "pv_production": [
                0.0,
                2.5,
                5.0,
            ],
            "consumption": [
                0.5,
                1.5,
                2.5,
            ],
        }
    )

    pv_scale, consumption_scale = (
        compute_normalization_scales(training_data)
    )

    assert pv_scale == pytest.approx(5.0)
    assert consumption_scale == pytest.approx(2.5)


def test_normalization_scales_have_safe_minimum():
    training_data = pd.DataFrame(
        {
            "pv_production": [
                0.0,
                0.2,
            ],
            "consumption": [
                0.1,
                0.5,
            ],
        }
    )

    pv_scale, consumption_scale = (
        compute_normalization_scales(training_data)
    )

    assert pv_scale == pytest.approx(1.0)
    assert consumption_scale == pytest.approx(1.0)


def test_empty_training_data_is_rejected():
    training_data = pd.DataFrame(
        columns=[
            "pv_production",
            "consumption",
        ]
    )

    with pytest.raises(ValueError):
        compute_normalization_scales(training_data)