"""Création des graphiques d'entraînement et d'évaluation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TRAINING_PATH = Path("results/training_metrics.csv")
EVALUATION_PATH = Path("results/evaluation_results.csv")
FIGURES_DIRECTORY = Path("results/figures")

POLICY_LABELS = {
    "idle": "Batterie inactive",
    "rule_based": "Règles simples",
    "dqn": "DQN",
}

POLICY_COLORS = ["gray", "orange", "royalblue"]


def add_value_labels(axis, decimals: int = 2) -> None:
    """Affiche la valeur numérique au-dessus de chaque barre."""
    for bar in axis.patches:
        value = bar.get_height()
        axis.annotate(
            f"{value:.{decimals}f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def create_training_figure(training: pd.DataFrame) -> None:
    """Trace les récompenses et epsilon pendant l'entraînement."""
    figure, first_axis = plt.subplots(figsize=(10, 6))

    first_axis.plot(
        training["episode"],
        training["training_reward"],
        marker="o",
        markersize=4,
        label="Entraînement",
    )
    first_axis.plot(
        training["episode"],
        training["validation_reward"],
        marker="s",
        markersize=4,
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

    figure.suptitle("Évolution de l’apprentissage du DQN")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIRECTORY / "training_progress.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def create_evaluation_figure(evaluation: pd.DataFrame) -> None:
    """Compare les politiques sur les principales métriques."""
    evaluation = evaluation.copy()
    evaluation["strategy"] = evaluation["policy"].map(POLICY_LABELS)

    metrics = [
        (
            "grid_import_kwh",
            "Importation du réseau",
            "Énergie (kWh)",
            2,
        ),
        (
            "unmet_demand_kwh",
            "Demande non satisfaite",
            "Énergie (kWh)",
            2,
        ),
        (
            "autonomy_rate_percent",
            "Autonomie énergétique",
            "Pourcentage (%)",
            2,
        ),
        (
            "equivalent_battery_cycles",
            "Cycles équivalents de batterie",
            "Nombre de cycles",
            2,
        ),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10),
    )

    for axis, (column, title, ylabel, decimals) in zip(
        axes.flat,
        metrics,
    ):
        axis.bar(
            evaluation["strategy"],
            evaluation[column],
            color=POLICY_COLORS,
        )
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", labelrotation=10)
        axis.grid(axis="y", alpha=0.3)
        add_value_labels(axis, decimals)

    figure.suptitle(
        "Comparaison des stratégies sur les données de test",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIRECTORY / "policy_comparison.png",
        dpi=200,
        bbox_inches="tight",
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