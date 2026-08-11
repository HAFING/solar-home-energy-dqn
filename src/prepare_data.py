"""Prépare les données 2023 à l'échelle d'une maison.

Ce script lit la feuille « 2023 data » du fichier Dataset.xlsx (source Zenodo,
système énergétique agrégé), conserve la forme temporelle horaire mais ramène
la consommation à 4 500 kWh/an et la production photovoltaïque à un système
de 5 kW crête. Le résultat est écrit dans data/processed_energy.csv et sert
d'entrée à l'environnement du DQN.
"""

from pathlib import Path

import pandas as pd
import os


print("Répertoire courant :", os.getcwd())

RAW_PATH = Path("data/Dataset.xlsx")
OUTPUT_PATH = Path("data/processed_energy.csv")
ANNUAL_LOAD_KWH = 4500.0
PV_PEAK_KW = 5.0


def prepare_energy_data() -> pd.DataFrame:
    """Lit, nettoie et met à l'échelle les données énergétiques"""
    raw = pd.read_excel(
        RAW_PATH,
        sheet_name="2023 data",
        usecols=["Date", "PV generation (kW)", "Consumption (kW)"],
    )

    # Typage robuste : toute valeur illisible devient NaN puis est supprimée.
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["PV generation (kW)"] = pd.to_numeric(
        raw["PV generation (kW)"], errors="coerce"
    )
    raw["Consumption (kW)"] = pd.to_numeric(
        raw["Consumption (kW)"], errors="coerce"
    )

    # Supprime les lignes parasites (fin de feuille), trie l'ordre horaire,
    # retire d'éventuels doublons d'horodatage et borne les valeurs à >= 0.
    data = raw.dropna().copy()
    data = data.sort_values("Date").drop_duplicates("Date")
    data[["PV generation (kW)", "Consumption (kW)"]] = data[
        ["PV generation (kW)", "Consumption (kW)"]
    ].clip(lower=0)

    # Conserver le profil de consommation et obtenir 4 500 kWh/an.
    consumption = data["Consumption (kW)"]
    data["load_kwh"] = consumption / consumption.sum() * ANNUAL_LOAD_KWH

    # Conserver le profil solaire et obtenir un pic de 5kW.
    pv = data["PV generation (kW)"]
    if pv.max() <= 0:
        raise ValueError("La production photovoltaïque maximale doit être positive.")
    data["pv_kwh"] = pv / pv.max() * PV_PEAK_KW

    result = data.rename(columns={"Date": "timestamp"})[
        ["timestamp", "load_kwh", "pv_kwh"]
    ].reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    return result


if __name__ == "__main__":
    prepared = prepare_energy_data()
    print("Fichier :", OUTPUT_PATH)
    print("Lignes :", len(prepared))
    print("Période :", prepared.timestamp.min(), "->", prepared.timestamp.max())
    print("Consommation annuelle :", prepared.load_kwh.sum(), "kWh")
    print("Pic PV :", prepared.pv_kwh.max(), "kW")
