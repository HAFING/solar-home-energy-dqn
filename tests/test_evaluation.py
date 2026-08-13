import numpy as np
import pandas as pd

from evaluate import (
    evaluate_policy,
    idle_policy,
    rule_based_policy,
)
from src.environment.energy_environment import EnergyEnvironment


def create_test_environment():
    data = pd.DataFrame(
        {
            "pv_production": [
                2.0,
                0.0,
                0.0,
                2.0,
            ],
            "consumption": [
                0.5,
                1.0,
                1.0,
                0.5,
            ],
            "grid_available": [
                1,
                0,
                1,
                1,
            ],
        }
    )

    return EnergyEnvironment(data)


def test_rule_based_policy_charges_then_discharges():
    environment = create_test_environment()
    state = environment.reset()

    first_action = rule_based_policy(state, environment)

    assert first_action == EnergyEnvironment.CHARGE

    state, _, _, _ = environment.step(first_action)

    second_action = rule_based_policy(state, environment)

    assert second_action == EnergyEnvironment.DISCHARGE


def test_evaluate_idle_policy_returns_expected_metrics():
    environment = create_test_environment()

    results = evaluate_policy(
        policy_name="idle",
        environment=environment,
        action_selector=idle_policy,
    )

    assert results["policy"] == "idle"
    assert results["battery_discharge_kwh"] == 0
    assert results["battery_charge_kwh"] == 0
    assert 0 <= results["demand_satisfaction_percent"] <= 100
    assert results["outage_hours"] == 1