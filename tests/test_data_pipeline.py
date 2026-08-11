from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    REQUIRED_COLUMNS,
    chronological_split,
    load_energy_data,
)

PROCESSED_PATH = Path("data/processed_energy.csv")

EXPECTED_ROWS = 8760
EXPECTED_ANNUAL_LOAD_KWH = 4500.0
EXPECTED_PV_PEAK_KW = 5.0


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    """Charge une fois la série préparée pour tous les tests."""
    if not PROCESSED_PATH.exists():
        pytest.skip(
            "data/processed_energy.csv absent : exécuter d'abord "
            "scripts/prepare_data.py."
        )
    return load_energy_data(PROCESSED_PATH)


def test_nombre_de_lignes(data: pd.DataFrame) -> None:
    """La série doit contenir 8 760 heures (une année complète)."""
    assert len(data) == EXPECTED_ROWS


def test_colonnes_exactes(data: pd.DataFrame) -> None:
    """Seules les trois colonnes attendues doivent être présentes."""
    assert list(data.columns) == REQUIRED_COLUMNS


def test_ordre_horaire(data: pd.DataFrame) -> None:
    """Les horodatages sont strictement croissants et espacés d'une heure."""
    assert data["timestamp"].is_monotonic_increasing
    assert not data["timestamp"].duplicated().any()
    deltas = data["timestamp"].diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(hours=1)


def test_absence_de_nan(data: pd.DataFrame) -> None:
    """Aucune valeur manquante n'est tolérée."""
    assert not data.isna().any().any()


def test_non_negativite(data: pd.DataFrame) -> None:
    """La consommation et la production restent positives ou nulles."""
    assert (data["load_kwh"] >= 0).all()
    assert (data["pv_kwh"] >= 0).all()


def test_echelle_consommation_annuelle(data: pd.DataFrame) -> None:
    """La consommation totale vaut 4 500 kWh sur l'année."""
    assert data["load_kwh"].sum() == pytest.approx(EXPECTED_ANNUAL_LOAD_KWH, rel=1e-6)


def test_echelle_pic_pv(data: pd.DataFrame) -> None:
    """Le pic de production photovoltaïque vaut 5 kW."""
    assert data["pv_kwh"].max() == pytest.approx(EXPECTED_PV_PEAK_KW, rel=1e-6)


def test_decoupage_proportions(data: pd.DataFrame) -> None:
    """Le découpage respecte 70 % / 15 % / 15 % et couvre toute la série."""
    train, validation, test = chronological_split(data)
    n = len(data)
    assert len(train) == int(0.70 * n)
    assert len(validation) == int(0.85 * n) - int(0.70 * n)
    assert len(train) + len(validation) + len(test) == n


def test_decoupage_ordre_chronologique(data: pd.DataFrame) -> None:
    """Les trois ensembles se suivent dans le temps, sans chevauchement."""
    train, validation, test = chronological_split(data)
    assert train["timestamp"].max() < validation["timestamp"].min()
    assert validation["timestamp"].max() < test["timestamp"].min()


def test_load_energy_data_rejette_valeurs_negatives(tmp_path: Path) -> None:
    """Le chargeur lève une erreur si une valeur négative est présente."""
    bad = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="h"),
            "load_kwh": [1.0, -0.5, 2.0],
            "pv_kwh": [0.0, 1.0, 2.0],
        }
    )
    csv_path = tmp_path / "bad.csv"
    bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_energy_data(csv_path)


def test_load_energy_data_rejette_colonnes_manquantes(tmp_path: Path) -> None:
    """Le chargeur lève une erreur si une colonne obligatoire manque."""
    bad = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="h"),
            "load_kwh": [1.0, 0.5, 2.0],
        }
    )
    csv_path = tmp_path / "bad.csv"
    bad.to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_energy_data(csv_path)
