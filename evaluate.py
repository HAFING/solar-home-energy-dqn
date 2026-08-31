"""Évaluation du DQN et comparaison avec des politiques de référence."""

import csv
from pathlib import Path
from typing import Callable

from src.agent import DQNAgent
from src.environment.energy_environment import EnergyEnvironment
from src.integration_data import (
    compute_normalization_scales,
    load_environment_data,
    split_environment_data,
)


MODEL_PATH = Path("models/best_dqn.pt")
RESULTS_PATH = Path("results/evaluation_results.csv")


def idle_policy(
    state,
    environment: EnergyEnvironment,
) -> int:
    """Ne charge et ne décharge jamais la batterie."""

    return EnergyEnvironment.IDLE


def rule_based_policy(
    state,
    environment: EnergyEnvironment,
) -> int:
    """Applique une stratégie simple de gestion de la batterie.

    La politique :
    - charge la batterie lorsqu'il existe un surplus solaire ;
    - décharge principalement pendant une coupure ;
    - conserve la batterie lorsque le réseau fonctionne.
    """

    row = environment.data.iloc[
        environment.current_step
    ]

    pv = float(
        row["pv_production"]
    )

    consumption = float(
        row["consumption"]
    )

    grid_available = int(
        row["grid_available"]
    )

    if (
        pv > consumption
        and environment.soc < 1.0
    ):
        return EnergyEnvironment.CHARGE

    if (
        consumption > pv
        and grid_available == 0
        and environment.soc > 0.0
    ):
        return EnergyEnvironment.DISCHARGE

    return EnergyEnvironment.IDLE


def evaluate_policy(
    policy_name: str,
    environment: EnergyEnvironment,
    action_selector: Callable,
) -> dict:
    """Évalue une politique sur un épisode complet."""

    state, _ = environment.reset()
    done = False

    total_reward = 0.0
    total_grid_import = 0.0
    total_unmet_demand = 0.0
    total_battery_charge = 0.0
    total_battery_discharge = 0.0

    action_counts = {
        EnergyEnvironment.IDLE: 0,
        EnergyEnvironment.CHARGE: 0,
        EnergyEnvironment.DISCHARGE: 0,
    }

    while not done:
        action = action_selector(
            state,
            environment,
        )

        action_counts[action] += 1

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(action)

        done = terminated or truncated

        total_reward += reward
        total_grid_import += info[
            "grid_import"
        ]
        total_unmet_demand += info[
            "unmet_demand"
        ]
        total_battery_charge += info[
            "battery_charge"
        ]
        total_battery_discharge += info[
            "battery_discharge"
        ]

        state = next_state

    total_consumption = float(
        environment.data[
            "consumption"
        ].sum()
    )

    autonomy_rate = (
        100
        * (
            1
            - total_grid_import
            / total_consumption
        )
        if total_consumption > 0
        else 0.0
    )

    demand_satisfaction_rate = (
        100
        * (
            1
            - total_unmet_demand
            / total_consumption
        )
        if total_consumption > 0
        else 0.0
    )

    equivalent_battery_cycles = (
        total_battery_discharge
        / environment.BATTERY_CAPACITY
    )

    outage_hours = int(
        (
            environment.data[
                "grid_available"
            ]
            == 0
        ).sum()
    )

    return {
        "policy": policy_name,
        "reward": round(
            total_reward,
            6,
        ),
        "grid_import_kwh": round(
            total_grid_import,
            6,
        ),
        "unmet_demand_kwh": round(
            total_unmet_demand,
            6,
        ),
        "battery_charge_kwh": round(
            total_battery_charge,
            6,
        ),
        "battery_discharge_kwh": round(
            total_battery_discharge,
            6,
        ),
        "equivalent_battery_cycles": round(
            equivalent_battery_cycles,
            6,
        ),
        "autonomy_rate_percent": round(
            autonomy_rate,
            4,
        ),
        "demand_satisfaction_percent": round(
            demand_satisfaction_rate,
            4,
        ),
        "final_soc": round(
            environment.soc,
            6,
        ),
        "outage_hours": outage_hours,
        "idle_actions": action_counts[
            EnergyEnvironment.IDLE
        ],
        "charge_actions": action_counts[
            EnergyEnvironment.CHARGE
        ],
        "discharge_actions": action_counts[
            EnergyEnvironment.DISCHARGE
        ],
    }


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    """Enregistre les résultats comparatifs dans un CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)


def main() -> None:
    """Lance l'évaluation sur les données de test."""

    complete_data = load_environment_data(
        seed=42
    )

    (
        train_data,
        _,
        test_data,
    ) = split_environment_data(
        complete_data
    )

    pv_scale, consumption_scale = (
        compute_normalization_scales(
            train_data
        )
    )

    idle_results = evaluate_policy(
        policy_name="idle",
        environment=EnergyEnvironment(
            test_data,
            pv_scale=pv_scale,
            consumption_scale=consumption_scale,
        ),
        action_selector=idle_policy,
    )

    rule_results = evaluate_policy(
        policy_name="rule_based",
        environment=EnergyEnvironment(
            test_data,
            pv_scale=pv_scale,
            consumption_scale=consumption_scale,
        ),
        action_selector=rule_based_policy,
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. "
            "Exécutez d'abord train.py."
        )

    agent = DQNAgent(
        state_size=6,
        action_size=3,
        device="cpu",
    )

    agent.load(MODEL_PATH)

    dqn_results = evaluate_policy(
        policy_name="dqn",
        environment=EnergyEnvironment(
            test_data,
            pv_scale=pv_scale,
            consumption_scale=consumption_scale,
        ),
        action_selector=(
            lambda state, environment:
            agent.act(
                state,
                explore=False,
            )
        ),
    )

    results = [
        idle_results,
        rule_results,
        dqn_results,
    ]

    save_results(
        results,
        RESULTS_PATH,
    )

    print(
        "\nÉVALUATION SUR LES DONNÉES DE TEST"
    )
    print(
        "=" * 74
    )

    for result in results:
        print(
            f"{result['policy']:12} | "
            f"récompense : "
            f"{result['reward']:9.2f} | "
            f"import : "
            f"{result['grid_import_kwh']:8.2f} kWh | "
            f"non satisfaite : "
            f"{result['unmet_demand_kwh']:6.2f} kWh | "
            f"autonomie : "
            f"{result['autonomy_rate_percent']:6.2f}% | "
            f"cycles : "
            f"{result['equivalent_battery_cycles']:6.2f}"
        )

    print(
        "=" * 74
    )
    print(
        "Résultats enregistrés dans :",
        RESULTS_PATH,
    )


if __name__ == "__main__":
    main()