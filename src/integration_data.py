"""Préparation des données destinées à l'environnement énergétique.

Ce module adapte les données nettoyées aux colonnes attendues par
EnergyEnvironment, simule les coupures du réseau et réalise un découpage
chronologique en ensembles d'entraînement, de validation et de test.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_PATH = Path("data/processed_energy.csv")


def simulate_grid_availability(
    number_of_hours: int,
    outage_start_probability: float = 0.02,
    min_outage_hours: int = 1,
    max_outage_hours: int = 4,
    seed: int = 42,
) -> np.ndarray:
    """Simule la disponibilité horaire du réseau électrique.

    La valeur 1 signifie que le réseau est disponible.
    La valeur 0 représente une coupure.
    """

    if number_of_hours <= 0:
        raise ValueError("number_of_hours doit être strictement positif.")

    if not 0 <= outage_start_probability <= 1:
        raise ValueError(
            "outage_start_probability doit être comprise entre 0 et 1."
        )

    if min_outage_hours <= 0:
        raise ValueError("min_outage_hours doit être strictement positif.")

    if max_outage_hours < min_outage_hours:
        raise ValueError(
            "max_outage_hours doit être supérieur ou égal à min_outage_hours."
        )

    rng = np.random.default_rng(seed)
    grid_available = np.ones(number_of_hours, dtype=np.int8)

    hour = 0

    while hour < number_of_hours:
        outage_starts = rng.random() < outage_start_probability

        if outage_starts:
            duration = int(
                rng.integers(
                    min_outage_hours,
                    max_outage_hours + 1,
                )
            )

            outage_end = min(hour + duration, number_of_hours)
            grid_available[hour:outage_end] = 0
            hour = outage_end
        else:
            hour += 1

    return grid_available


def load_environment_data(
    csv_path: str | Path = DEFAULT_DATA_PATH,
    outage_start_probability: float = 0.02,
    min_outage_hours: int = 1,
    max_outage_hours: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """Charge et adapte les données pour EnergyEnvironment."""

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Le fichier de données est introuvable : {csv_path}"
        )

    data = pd.read_csv(csv_path, parse_dates=["timestamp"])

    required_columns = {
        "timestamp",
        "load_kwh",
        "pv_kwh",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Colonnes manquantes : "
            + ", ".join(sorted(missing_columns))
        )

    data = data[
        ["timestamp", "load_kwh", "pv_kwh"]
    ].copy()

    data = data.dropna()
    data = data.sort_values("timestamp")
    data = data.drop_duplicates("timestamp")
    data = data.reset_index(drop=True)

    data["load_kwh"] = data["load_kwh"].clip(lower=0)
    data["pv_kwh"] = data["pv_kwh"].clip(lower=0)

    data = data.rename(
        columns={
            "load_kwh": "consumption",
            "pv_kwh": "pv_production",
        }
    )

    data["grid_available"] = simulate_grid_availability(
        number_of_hours=len(data),
        outage_start_probability=outage_start_probability,
        min_outage_hours=min_outage_hours,
        max_outage_hours=max_outage_hours,
        seed=seed,
    )

    return data[
        [
            "timestamp",
            "consumption",
            "pv_production",
            "grid_available",
        ]
    ]


def split_environment_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe une année de 365 jours dans l'ordre chronologique.

    Répartition :
    - entraînement : 255 jours ;
    - validation : 55 jours ;
    - test : 55 jours.
    """

    required_hours = 365 * 24

    if len(data) < required_hours:
        raise ValueError(
            "Le découpage nécessite au moins 8 760 observations horaires."
        )

    data = data.iloc[:required_hours].copy()

    train_end = 255 * 24
    validation_end = train_end + (55 * 24)

    train_data = data.iloc[:train_end].reset_index(drop=True)

    validation_data = data.iloc[
        train_end:validation_end
    ].reset_index(drop=True)

    test_data = data.iloc[
        validation_end:
    ].reset_index(drop=True)

    return train_data, validation_data, test_data


if __name__ == "__main__":
    environment_data = load_environment_data()

    train_data, validation_data, test_data = (
        split_environment_data(environment_data)
    )

    outage_hours = int(
        (environment_data["grid_available"] == 0).sum()
    )

    print("Observations totales :", len(environment_data))
    print("Données d'entraînement :", len(train_data))
    print("Données de validation :", len(validation_data))
    print("Données de test :", len(test_data))
    print("Heures de coupure simulées :", outage_hours)
    print(
        "Taux de disponibilité du réseau :",
        f"{environment_data['grid_available'].mean():.2%}",
    )