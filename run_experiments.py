"""Exécute plusieurs expériences DQN avec des graines différentes."""

import argparse
from pathlib import Path

import pandas as pd

from evaluate import (
    evaluate_policy,
    idle_policy,
    rule_based_policy,
    save_results,
)
from src.agent import DQNAgent
from src.environment.energy_environment import (
    EnergyEnvironment,
)
from src.integration_data import (
    compute_normalization_scales,
    load_environment_data,
    split_environment_data,
)
from train import train


DEFAULT_SEEDS = [
    42,
    123,
    2026,
]

MODEL_DIRECTORY = Path(
    "models/experiments"
)

RESULTS_DIRECTORY = Path(
    "results/experiments"
)

SUMMARY_METRICS = [
    "reward",
    "grid_import_kwh",
    "unmet_demand_kwh",
    "battery_throughput_kwh",
    "battery_degradation_cost",
    "equivalent_battery_cycles",
    "autonomy_rate_percent",
    "demand_satisfaction_percent",
]


def create_test_environment(
    test_data,
    pv_scale: float,
    consumption_scale: float,
) -> EnergyEnvironment:
    """Crée un environnement de test normalisé avec train."""

    return EnergyEnvironment(
        test_data,
        pv_scale=pv_scale,
        consumption_scale=consumption_scale,
    )


def evaluate_trained_model(
    seed: int,
    model_path: Path,
) -> list[dict]:
    """Évalue les trois politiques pour une graine."""

    complete_data = load_environment_data(
        seed=seed
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
        environment=create_test_environment(
            test_data,
            pv_scale,
            consumption_scale,
        ),
        action_selector=idle_policy,
    )

    rule_results = evaluate_policy(
        policy_name="rule_based",
        environment=create_test_environment(
            test_data,
            pv_scale,
            consumption_scale,
        ),
        action_selector=rule_based_policy,
    )

    agent = DQNAgent(
        state_size=6,
        action_size=3,
        device="cpu",
    )

    agent.load(
        model_path
    )

    dqn_results = evaluate_policy(
        policy_name="dqn",
        environment=create_test_environment(
            test_data,
            pv_scale,
            consumption_scale,
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

    for result in results:
        result["seed"] = seed

    return results


def summarize_results(
    results: list[dict],
) -> pd.DataFrame:
    """Calcule moyenne et écart-type par politique."""

    if not results:
        raise ValueError(
            "Aucun résultat à résumer."
        )

    results_frame = pd.DataFrame(
        results
    )

    missing_columns = set(
        SUMMARY_METRICS
    ).difference(
        results_frame.columns
    )

    if missing_columns:
        raise ValueError(
            "Métriques manquantes : "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    summary = (
        results_frame.groupby(
            "policy"
        )[SUMMARY_METRICS]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    summary.columns = [
        (
            column
            if isinstance(
                column,
                str,
            )
            else "_".join(
                part
                for part in column
                if part
            )
        )
        for column in summary.columns
    ]

    summary = (
        summary.fillna(
            0.0
        )
        .round(
            6
        )
    )

    return summary


def run_experiments(
    seeds: list[int],
    episodes: int,
) -> dict:
    """Entraîne et évalue une expérience par graine."""

    if episodes <= 0:
        raise ValueError(
            "Le nombre d'épisodes doit être positif."
        )

    if not seeds:
        raise ValueError(
            "Au moins une graine est nécessaire."
        )

    if len(seeds) != len(set(seeds)):
        raise ValueError(
            "Les graines doivent être uniques."
        )

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []
    manifest = []

    for seed in seeds:
        print()
        print(
            "=" * 72
        )
        print(
            f"EXPÉRIENCE — GRAINE {seed}"
        )
        print(
            "=" * 72
        )

        model_path = (
            MODEL_DIRECTORY
            / f"dqn_seed_{seed}.pt"
        )

        training_metrics_path = (
            RESULTS_DIRECTORY
            / f"training_seed_{seed}.csv"
        )

        evaluation_results_path = (
            RESULTS_DIRECTORY
            / f"evaluation_seed_{seed}.csv"
        )

        history = train(
            episodes=episodes,
            seed=seed,
            model_path=model_path,
            metrics_path=training_metrics_path,
        )

        best_validation_reward = max(
            row["validation_reward"]
            for row in history
        )

        seed_results = evaluate_trained_model(
            seed=seed,
            model_path=model_path,
        )

        for result in seed_results:
            result[
                "best_validation_reward"
            ] = round(
                best_validation_reward,
                6,
            )

        save_results(
            seed_results,
            evaluation_results_path,
        )

        all_results.extend(
            seed_results
        )

        manifest.append(
            {
                "seed": seed,
                "episodes": episodes,
                "best_validation_reward": round(
                    best_validation_reward,
                    6,
                ),
                "model_path": str(
                    model_path
                ),
                "training_metrics_path": str(
                    training_metrics_path
                ),
                "evaluation_results_path": str(
                    evaluation_results_path
                ),
            }
        )

    all_results_path = (
        RESULTS_DIRECTORY
        / "all_evaluation_results.csv"
    )

    save_results(
        all_results,
        all_results_path,
    )

    summary = summarize_results(
        all_results
    )

    summary_path = (
        RESULTS_DIRECTORY
        / "summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    manifest_frame = pd.DataFrame(
        manifest
    )

    manifest_path = (
        RESULTS_DIRECTORY
        / "manifest.csv"
    )

    manifest_frame.to_csv(
        manifest_path,
        index=False,
    )

    best_experiment = max(
        manifest,
        key=lambda row: row[
            "best_validation_reward"
        ],
    )

    print()
    print(
        "=" * 72
    )
    print(
        "SYNTHÈSE DES EXPÉRIENCES"
    )
    print(
        "=" * 72
    )
    print(
        summary.to_string(
            index=False
        )
    )
    print()
    print(
        "Meilleure graine selon la validation :",
        best_experiment["seed"],
    )
    print(
        "Meilleure récompense de validation :",
        best_experiment[
            "best_validation_reward"
        ],
    )
    print(
        "Modèle correspondant :",
        best_experiment[
            "model_path"
        ],
    )
    print(
        "Synthèse enregistrée dans :",
        summary_path,
    )

    return {
        "results": all_results,
        "summary": summary,
        "manifest": manifest,
        "best_experiment": best_experiment,
    }


def parse_arguments() -> argparse.Namespace:
    """Lit les arguments du terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Exécuter les expériences DQN "
            "avec plusieurs graines."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30,
        help=(
            "Nombre d'épisodes pour chaque graine."
        ),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=(
            "Liste des graines expérimentales."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    run_experiments(
        seeds=arguments.seeds,
        episodes=arguments.episodes,
    )