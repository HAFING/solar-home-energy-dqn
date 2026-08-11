
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ["timestamp", "load_kwh", "pv_kwh"]

# Proportions du découpage chronologique.
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15  # le test reçoit le reste (0,15).


def load_energy_data(path: str | Path) -> pd.DataFrame:
    """Charge et valide la série horaire préparée.

    Args:
        path: chemin vers le fichier processed_energy.csv.

    Returns:
        DataFrame trié par timestamp avec les colonnes timestamp, load_kwh
        et pv_kwh.

    Raises:
        FileNotFoundError: si le fichier est absent.
        ValueError: si les colonnes, les NaN ou des valeurs négatives sont
            incorrects.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    data = pd.read_csv(path, parse_dates=["timestamp"])

    # Les trois colonnes doivent être présentes.
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")
    data = data[REQUIRED_COLUMNS].copy()

    # Aucune valeur manquante ne doit entrer dans l'environnement
    if data.isna().any().any():
        raise ValueError("Le fichier contient des valeurs manquantes (NaN).")

    # La consommation et la production ne peuvent pas être négatives
    if (data["load_kwh"] < 0).any() or (data["pv_kwh"] < 0).any():
        raise ValueError("Valeurs négatives détectées dans load_kwh ou pv_kwh.")

    # Ordre horaire garanti, sans doublon d'horodatage
    data = data.sort_values("timestamp").reset_index(drop=True)
    if data["timestamp"].duplicated().any():
        raise ValueError("Horodatages en double détectés.")

    return data


def chronological_split(
 data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe la série en train / validation / test sans mélange.

    Le découpage respecte l'ordre du temps : les premières heures servent à
    l'entraînement, les suivantes à la validation, les dernières au test.

    Args:
        data: DataFrame trié renvoyé par load_energy_data().

    Returns:
        Un triplet (train, validation, test) de DataFrames disjoints et
        contigus dans le temps.
    """
    data = data.sort_values("timestamp").reset_index(drop=True)

    n = len(data)
    train_end = int(TRAIN_RATIO * n)
    validation_end = int((TRAIN_RATIO + VALIDATION_RATIO) * n)

    train = data.iloc[:train_end].reset_index(drop=True)
    validation = data.iloc[train_end:validation_end].reset_index(drop=True)
    test = data.iloc[validation_end:].reset_index(drop=True)

    return train, validation, test


if __name__ == "__main__":
    df = load_energy_data("data/processed_energy.csv")
    tr, va, te = chronological_split(df)
    print("Total :", len(df))
    print("Train :", len(tr), "Validation :", len(va), "Test :", len(te))
    print("Fin train :", tr.timestamp.max())
    print("Début validation :", va.timestamp.min())
    print("Début test :", te.timestamp.min())
