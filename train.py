"""Entraînement du DQN pour la gestion énergétique de la maison solaire."""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch

from src.agent import DQNAgent
from src.environment.energy_environment import EnergyEnvironment
from src.integration_data import (
    compute_normalization_scales,
    load_environment_data,
    split_environment_data,
)


MODEL_PATH = Path("models/best_dqn.pt")
METRICS_PATH = Path("results/training_metrics.csv")


def set_random_seeds(seed: int) -> None:
    """Rend l'expérience aussi reproductible que possible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training_episode(
    environment: EnergyEnvironment,
    agent: DQNAgent,
) -> dict:
    """Exécute un épisode d'apprentissage complet."""

    state, _ = environment.reset()
    done = False

    total_reward = 0.0
    total_loss = 0.0
    optimization_steps = 0
    total_grid_import = 0.0
    total_unmet_demand = 0.0
    total_battery_discharge = 0.0

    while not done:
        action = agent.act(
            state,
            explore=True,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(action)

        done = terminated or truncated

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

        loss = agent.optimize_model()

        if loss is not None:
            total_loss += loss
            optimization_steps += 1

        total_reward += reward
        total_grid_import += info["grid_import"]
        total_unmet_demand += info["unmet_demand"]
        total_battery_discharge += info[
            "battery_discharge"
        ]

        state = next_state

    average_loss = (
        total_loss / optimization_steps
        if optimization_steps > 0
        else 0.0
    )

    return {
        "reward": total_reward,
        "average_loss": average_loss,
        "grid_import": total_grid_import,
        "unmet_demand": total_unmet_demand,
        "battery_discharge": total_battery_discharge,
        "final_soc": environment.soc,
    }


def evaluate_agent(
    environment: EnergyEnvironment,
    agent: DQNAgent,
) -> dict:
    """Évalue l'agent sans exploration et sans apprentissage."""

    state, _ = environment.reset()
    done = False

    total_reward = 0.0
    total_grid_import = 0.0
    total_unmet_demand = 0.0
    total_battery_discharge = 0.0

    while not done:
        action = agent.act(
            state,
            explore=False,
        )

        (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        ) = environment.step(action)

        done = terminated or truncated

        total_reward += reward
        total_grid_import += info["grid_import"]
        total_unmet_demand += info["unmet_demand"]
        total_battery_discharge += info[
            "battery_discharge"
        ]

        state = next_state

    return {
        "reward": total_reward,
        "grid_import": total_grid_import,
        "unmet_demand": total_unmet_demand,
        "battery_discharge": total_battery_discharge,
        "final_soc": environment.soc,
    }


def save_metrics(
    metrics: list[dict],
    output_path: Path,
) -> None:
    """Enregistre les métriques d'entraînement dans un CSV."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not metrics:
        return

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=metrics[0].keys(),
        )

        writer.writeheader()
        writer.writerows(metrics)


def train(
    episodes: int = 10,
    seed: int = 42,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> list[dict]:
    """Entraîne le DQN et sauvegarde le meilleur modèle."""

    if episodes <= 0:
        raise ValueError(
            "Le nombre d'épisodes doit être positif."
        )

    set_random_seeds(seed)

    complete_data = load_environment_data(
        seed=seed
    )

    (
        train_data,
        validation_data,
        _,
    ) = split_environment_data(
        complete_data
    )

    pv_scale, consumption_scale = (
        compute_normalization_scales(
            train_data
        )
    )

    training_environment = EnergyEnvironment(
        train_data,
        pv_scale=pv_scale,
        consumption_scale=consumption_scale,
    )

    validation_environment = EnergyEnvironment(
        validation_data,
        pv_scale=pv_scale,
        consumption_scale=consumption_scale,
    )

    agent = DQNAgent(
        state_size=6,
        action_size=3,
        hidden_size=64,
        buffer_capacity=50_000,
        batch_size=64,
        gamma=0.99,
        learning_rate=1e-3,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.99995,
        target_update_freq=250,
    )

    print(
        "Appareil utilisé :",
        agent.device,
    )
    print(
        "Épisodes :",
        episodes,
    )
    print(
        "Heures d'entraînement :",
        len(train_data),
    )
    print(
        "Heures de validation :",
        len(validation_data),
    )
    print(
        "Échelle solaire issue de train :",
        f"{pv_scale:.6f}",
    )
    print(
        "Échelle consommation issue de train :",
        f"{consumption_scale:.6f}",
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_validation_reward = float("-inf")
    history = []

    for episode in range(
        1,
        episodes + 1,
    ):
        training_results = run_training_episode(
            training_environment,
            agent,
        )

        validation_results = evaluate_agent(
            validation_environment,
            agent,
        )

        episode_metrics = {
            "episode": episode,
            "training_reward": round(
                training_results["reward"],
                6,
            ),
            "validation_reward": round(
                validation_results["reward"],
                6,
            ),
            "average_loss": round(
                training_results["average_loss"],
                6,
            ),
            "epsilon": round(
                agent.epsilon,
                6,
            ),
            "training_grid_import": round(
                training_results["grid_import"],
                6,
            ),
            "validation_grid_import": round(
                validation_results["grid_import"],
                6,
            ),
            "training_unmet_demand": round(
                training_results["unmet_demand"],
                6,
            ),
            "validation_unmet_demand": round(
                validation_results["unmet_demand"],
                6,
            ),
            "validation_battery_discharge": round(
                validation_results[
                    "battery_discharge"
                ],
                6,
            ),
            "validation_final_soc": round(
                validation_results["final_soc"],
                6,
            ),
        }

        history.append(episode_metrics)

        print(
            f"Épisode {episode:02d}/{episodes} | "
            f"récompense train : "
            f"{training_results['reward']:.2f} | "
            f"récompense validation : "
            f"{validation_results['reward']:.2f} | "
            f"epsilon : {agent.epsilon:.3f} | "
            f"demande non satisfaite : "
            f"{validation_results['unmet_demand']:.2f}"
        )

        if (
            validation_results["reward"]
            > best_validation_reward
        ):
            best_validation_reward = (
                validation_results["reward"]
            )

            agent.save(model_path)

            print(
                "Nouveau meilleur modèle enregistré dans",
                model_path,
            )

        save_metrics(
            history,
            metrics_path,
        )

    print(
        "Entraînement terminé."
    )
    print(
        "Meilleure récompense de validation :",
        f"{best_validation_reward:.2f}",
    )
    print(
        "Modèle :",
        model_path,
    )
    print(
        "Métriques :",
        metrics_path,
    )

    return history


def parse_arguments() -> argparse.Namespace:
    """Lit les paramètres fournis dans le terminal."""

    parser = argparse.ArgumentParser(
        description=(
            "Entraîner le DQN de gestion énergétique."
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Nombre d'épisodes d'entraînement.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()

    train(
        episodes=arguments.episodes,
        seed=arguments.seed,
    )