"""Interface Gradio de démonstration du projet énergétique DQN."""

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd


EVALUATION_PATH = Path("results/evaluation_results.csv")
TRAINING_FIGURE_PATH = Path(
    "results/figures/training_progress.png"
)
COMPARISON_FIGURE_PATH = Path(
    "results/figures/policy_comparison.png"
)

POLICY_LABELS = {
    "idle": "Batterie inactive",
    "rule_based": "Règles simples",
    "dqn": "DQN",
}

POLICY_IDS = {
    label: policy_id
    for policy_id, label in POLICY_LABELS.items()
}


def load_results() -> pd.DataFrame:
    """Charge les résultats d'évaluation enregistrés."""
    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {EVALUATION_PATH}. "
            "Exécutez d'abord evaluate.py."
        )

    return pd.read_csv(EVALUATION_PATH)


def display_results() -> pd.DataFrame:
    """Retourne une table lisible pour l'interface."""
    results = load_results().copy()

    results["Politique"] = results["policy"].map(
        POLICY_LABELS
    )

    columns = [
        "Politique",
        "reward",
        "grid_import_kwh",
        "unmet_demand_kwh",
        "autonomy_rate_percent",
        "equivalent_battery_cycles",
    ]

    table = results[columns].rename(
        columns={
            "reward": "Récompense",
            "grid_import_kwh": "Import réseau (kWh)",
            "unmet_demand_kwh": (
                "Demande non satisfaite (kWh)"
            ),
            "autonomy_rate_percent": "Autonomie (%)",
            "equivalent_battery_cycles": (
                "Cycles équivalents"
            ),
        }
    )

    return table.round(2)


def get_policy_result(policy_label: str) -> pd.Series:
    """Retourne les métriques de la politique sélectionnée."""
    policy_id = POLICY_IDS[policy_label]
    results = load_results()

    selected = results.loc[
        results["policy"] == policy_id
    ]

    if selected.empty:
        raise ValueError(
            f"Politique introuvable : {policy_label}"
        )

    return selected.iloc[0]


def policy_explanation(policy_label: str) -> str:
    """Construit l'analyse textuelle de la politique sélectionnée."""
    result = get_policy_result(policy_label)

    reward = float(result["reward"])
    grid_import = float(result["grid_import_kwh"])
    unmet_demand = float(result["unmet_demand_kwh"])
    autonomy = float(result["autonomy_rate_percent"])
    cycles = float(result["equivalent_battery_cycles"])

    specific_analysis = {
        "Batterie inactive": (
            "Cette référence ne pilote jamais la batterie. "
            "Elle permet de mesurer le gain apporté par les "
            "stratégies de gestion."
        ),
        "Règles simples": (
            "Cette stratégie conserve principalement la batterie "
            "lorsque le réseau est disponible et la décharge "
            "pendant les coupures. Elle privilégie donc la "
            "continuité de service et la durée de vie de la batterie."
        ),
        "DQN": (
            "Le DQN apprend une politique à partir des interactions "
            "état-action-récompense. Il réduit davantage "
            "l'importation du réseau et obtient la meilleure "
            "autonomie, mais utilise plus fréquemment la batterie."
        ),
    }

    return f"""
## {policy_label}

| Métrique | Valeur |
|---|---:|
| Récompense | {reward:.2f} |
| Import réseau | {grid_import:.2f} kWh |
| Demande non satisfaite | {unmet_demand:.2f} kWh |
| Autonomie énergétique | {autonomy:.2f} % |
| Cycles équivalents | {cycles:.2f} |

{specific_analysis[policy_label]}

L'autonomie mesure la part de la consommation ne dépendant pas du réseau.
La demande non satisfaite est présentée séparément afin d'évaluer la
continuité de service pendant les coupures.
"""


def create_policy_figure(policy_label: str):
    """Crée un graphique détaillé pour une politique."""
    result = get_policy_result(policy_label)

    metrics = [
        (
            "Import réseau",
            float(result["grid_import_kwh"]),
            "kWh",
        ),
        (
            "Demande non satisfaite",
            float(result["unmet_demand_kwh"]),
            "kWh",
        ),
        (
            "Autonomie",
            float(result["autonomy_rate_percent"]),
            "%",
        ),
        (
            "Cycles équivalents",
            float(result["equivalent_battery_cycles"]),
            "cycles",
        ),
    ]

    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
    )

    color = {
        "Batterie inactive": "gray",
        "Règles simples": "orange",
        "DQN": "royalblue",
    }[policy_label]

    for axis, (title, value, unit) in zip(
        axes.flat,
        metrics,
    ):
        axis.bar(
            [policy_label],
            [value],
            color=color,
        )
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis="y", alpha=0.3)
        axis.text(
            0,
            value,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    figure.suptitle(
        f"Indicateurs de la politique : {policy_label}",
        fontsize=14,
    )
    figure.tight_layout()

    return figure


def update_policy_view(policy_label: str):
    """Met à jour l'analyse et le graphique sélectionnés."""
    return (
        policy_explanation(policy_label),
        create_policy_figure(policy_label),
    )


def rerun_evaluation():
    """Relance l'évaluation officielle et recharge les résultats."""
    from evaluate import main as evaluate_main

    evaluate_main()

    return (
        display_results(),
        "Évaluation relancée avec succès.",
    )


def image_or_none(image_path: Path):
    """Retourne l'image si elle existe."""
    if image_path.exists():
        return str(image_path)

    return None


def build_demo() -> gr.Blocks:
    """Construit l'interface Gradio."""
    default_policy = "DQN"

    with gr.Blocks(
        title="Gestion énergétique solaire par DQN"
    ) as demo:
        gr.Markdown(
            """
# Gestion énergétique d’une maison solaire par DQN

Cette interface présente les résultats du projet de Reinforcement Learning.
Elle compare une batterie inactive, une stratégie à règles simples et
l’agent DQN entraîné manuellement avec PyTorch.
"""
        )

        with gr.Tab("Comparaison des politiques"):
            gr.Markdown(
                """
Les résultats ci-dessous proviennent de l'évaluation sur les données de test.
Le DQN vise à réduire la dépendance au réseau tout en tenant compte de
la demande non satisfaite et de l'usure de la batterie.
"""
            )

            results_table = gr.Dataframe(
                value=display_results(),
                interactive=False,
                label="Résultats de test",
            )

            refresh_button = gr.Button(
                "Relancer l'évaluation officielle"
            )

            refresh_status = gr.Markdown()

            refresh_button.click(
                fn=rerun_evaluation,
                outputs=[
                    results_table,
                    refresh_status,
                ],
            )

        with gr.Tab("Analyse d'une politique"):
            policy_selector = gr.Dropdown(
                choices=list(POLICY_IDS.keys()),
                value=default_policy,
                label="Politique à analyser",
            )

            analysis_markdown = gr.Markdown(
                value=policy_explanation(default_policy)
            )

            policy_plot = gr.Plot(
                value=create_policy_figure(default_policy),
                label="Indicateurs de la politique",
            )

            policy_selector.change(
                fn=update_policy_view,
                inputs=policy_selector,
                outputs=[
                    analysis_markdown,
                    policy_plot,
                ],
            )

        with gr.Tab("Graphiques du projet"):
            with gr.Row():
                gr.Image(
                    value=image_or_none(TRAINING_FIGURE_PATH),
                    label="Progression de l'entraînement",
                    interactive=False,
                )
                gr.Image(
                    value=image_or_none(COMPARISON_FIGURE_PATH),
                    label="Comparaison des politiques",
                    interactive=False,
                )

        with gr.Tab("Méthodologie"):
            gr.Markdown(
                """
## Méthodologie

- L'environnement respecte l'API Gymnasium.
- Le DQN est codé manuellement avec PyTorch.
- Les données sont séparées chronologiquement en entraînement,
  validation et test.
- Les échelles de normalisation sont calculées uniquement sur
  l'entraînement.
- Trois graines aléatoires sont utilisées : `42`, `123` et `2026`.
- Le meilleur modèle est sélectionné selon la récompense de validation.
- L'autonomie mesure la part de consommation non dépendante du réseau.
- La demande non satisfaite et les cycles équivalents sont suivis
  séparément pour évaluer la fiabilité et l'usure de la batterie.
"""
            )

    return demo


if __name__ == "__main__":
    build_demo().launch()