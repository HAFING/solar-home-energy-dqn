"""Création des graphiques d'entraînement et d'évaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRAINING_PATH = Path("results/training_metrics.csv")
EVALUATION_PATH = Path("results/evaluation_results.csv")
FIGURES_DIRECTORY = Path("results/figures")


def create_training_figure(training: pd.DataFrame) -> None:
    """Trace les récompenses et epsilon pendant l'entraînement."""

    figure, first_axis = plt.subplots(figsize=(10, 6))

    first_axis.plot(
        training["episode"],
        training["training_reward"],
        marker="o",
        label="Entraînement",
    )

    first_axis.plot(
        training["episode"],
        training["validation_reward"],
        marker="s",
        label="Validation",
    )

    first_axis.set_xlabel("Épisode")
    first_axis.set_ylabel("Récompense")
    first_axis.grid(alpha=0.3)
    first_axis.legend(loc="upper left")

    second_axis = first_axis.twinx()

    second_axis.plot(
        training["episode"],
        training["epsilon"],
        color="black",
        linestyle="--",
        label="Epsilon",
    )

    second_axis.set_ylabel("Epsilon")
    second_axis.set_ylim(0, 1.05)

    plt.title("Évolution de l’apprentissage du DQN")
    figure.tight_layout()

    figure.savefig(
        FIGURES_DIRECTORY / "training_progress.png",
        dpi=200,
    )

    plt.close(figure)


def create_evaluation_figure(evaluation: pd.DataFrame) -> None:
    """Compare les politiques sur les principales métriques."""

    labels = {
        "idle": "Batterie inactive",
        "rule_based": "Règles simples",
        "dqn": "DQN",
    }

    evaluation = evaluation.copy()

    evaluation["strategy"] = evaluation["policy"].map(labels)

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    axes[0].bar(
        evaluation["strategy"],
        evaluation["grid_import_kwh"],
        color=["gray", "orange", "royalblue"],
    )
    axes[0].set_title("Importation du réseau")
    axes[0].set_ylabel("Énergie (kWh)")

    axes[1].bar(
        evaluation["strategy"],
        evaluation["unmet_demand_kwh"],
        color=["gray", "orange", "royalblue"],
    )
    axes[1].set_title("Demande non satisfaite")
    axes[1].set_ylabel("Énergie (kWh)")

    axes[2].bar(
        evaluation["strategy"],
        evaluation["autonomy_rate_percent"],
        color=["gray", "orange", "royalblue"],
    )
    axes[2].set_title("Autonomie énergétique")
    axes[2].set_ylabel("Pourcentage (%)")

    for axis in axes:
        axis.tick_params(
            axis="x",
            labelrotation=20,
        )
        axis.grid(
            axis="y",
            alpha=0.3,
        )

    figure.suptitle(
        "Comparaison des stratégies sur les données de test"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIRECTORY / "policy_comparison.png",
        dpi=200,
    )

    plt.close(figure)


def main() -> None:
    """Charge les résultats et produit les deux graphiques."""

    if not TRAINING_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {TRAINING_PATH}"
        )

    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {EVALUATION_PATH}"
        )

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    training = pd.read_csv(TRAINING_PATH)
    evaluation = pd.read_csv(EVALUATION_PATH)

    create_training_figure(training)
    create_evaluation_figure(evaluation)

    print(
        "Graphique d'entraînement :",
        FIGURES_DIRECTORY / "training_progress.png",
    )

    print(
        "Graphique de comparaison :",
        FIGURES_DIRECTORY / "policy_comparison.png",
    )


if __name__ == "__main__":
    main()