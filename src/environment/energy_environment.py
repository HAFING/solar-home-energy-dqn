"""Environnement Gymnasium de gestion énergétique d'une maison solaire."""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class EnergyEnvironment(gym.Env):
    """Simule la gestion horaire d'une maison solaire avec batterie.

    État :
        [
            production photovoltaïque normalisée,
            consommation normalisée,
            état de charge de la batterie,
            disponibilité du réseau,
            sinus de l'heure,
            cosinus de l'heure,
        ]

    Actions :
        0 -> IDLE
        1 -> CHARGE
        2 -> DISCHARGE
    """

    metadata = {
        "render_modes": [],
    }

    IDLE = 0
    CHARGE = 1
    DISCHARGE = 2

    BATTERY_CAPACITY = 10.0
    INITIAL_SOC = 0.50
    MAX_BATTERY_POWER = 2.0
    BATTERY_EFFICIENCY = 0.95

    def __init__(
        self,
        data,
        pv_scale: float | None = None,
        consumption_scale: float | None = None,
        initial_soc: float = INITIAL_SOC,
    ):
        super().__init__()

        required_columns = {
            "pv_production",
            "consumption",
            "grid_available",
        }

        missing_columns = required_columns.difference(data.columns)

        if missing_columns:
            raise ValueError(
                "Colonnes manquantes : "
                + ", ".join(sorted(missing_columns))
            )

        if len(data) == 0:
            raise ValueError(
                "L'environnement nécessite au moins une observation."
            )

        if not 0 <= initial_soc <= 1:
            raise ValueError(
                "initial_soc doit être compris entre 0 et 1."
            )

        self.data = data.reset_index(drop=True).copy()
        self.initial_soc = float(initial_soc)

        inferred_pv_scale = max(
            float(self.data["pv_production"].max()),
            1.0,
        )

        inferred_consumption_scale = max(
            float(self.data["consumption"].max()),
            1.0,
        )

        self.pv_scale = (
            float(pv_scale)
            if pv_scale is not None
            else inferred_pv_scale
        )

        self.consumption_scale = (
            float(consumption_scale)
            if consumption_scale is not None
            else inferred_consumption_scale
        )

        if self.pv_scale <= 0:
            raise ValueError(
                "pv_scale doit être strictement positif."
            )

        if self.consumption_scale <= 0:
            raise ValueError(
                "consumption_scale doit être strictement positif."
            )

        # Alias conservés pour faciliter la lecture et la compatibilité.
        self.pv_max = self.pv_scale
        self.consumption_max = self.consumption_scale

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, 0.0, -1.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.render_mode = None
        self.current_step = 0
        self.soc = self.initial_soc

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        """Réinitialise l'épisode et renvoie l'état initial."""

        super().reset(seed=seed)

        self.current_step = 0
        reset_soc = self.initial_soc

        if options is not None and "initial_soc" in options:
            reset_soc = float(options["initial_soc"])

            if not 0 <= reset_soc <= 1:
                raise ValueError(
                    "Le SOC initial doit être compris entre 0 et 1."
                )

        self.soc = reset_soc

        observation = self._get_state()
        info = {}

        return observation, info

    def _get_state(self) -> np.ndarray:
        """Construit l'observation correspondant à l'heure actuelle."""

        row = self.data.iloc[self.current_step]

        pv = float(row["pv_production"])
        consumption = float(row["consumption"])
        grid_available = float(row["grid_available"])

        pv_normalized = np.clip(
            pv / self.pv_scale,
            0.0,
            1.0,
        )

        consumption_normalized = np.clip(
            consumption / self.consumption_scale,
            0.0,
            1.0,
        )

        if "timestamp" in self.data.columns:
            timestamp = row["timestamp"]
            hour = int(
                getattr(
                    timestamp,
                    "hour",
                    self.current_step % 24,
                )
            )
        else:
            hour = self.current_step % 24

        sin_hour = math.sin(
            2 * math.pi * hour / 24
        )

        cos_hour = math.cos(
            2 * math.pi * hour / 24
        )

        return np.array(
            [
                pv_normalized,
                consumption_normalized,
                self.soc,
                grid_available,
                sin_hour,
                cos_hour,
            ],
            dtype=np.float32,
        )

    def step(self, action):
        """Applique une action et avance la simulation d'une heure."""

        if not self.action_space.contains(action):
            raise ValueError(
                f"Action invalide : {action}"
            )

        action = int(action)
        row = self.data.iloc[self.current_step]

        pv = float(row["pv_production"])
        consumption = float(row["consumption"])
        grid_available = int(row["grid_available"])

        battery_energy = (
            self.soc * self.BATTERY_CAPACITY
        )

        solar_used = min(
            pv,
            consumption,
        )

        surplus = max(
            pv - consumption,
            0.0,
        )

        deficit = max(
            consumption - pv,
            0.0,
        )

        battery_charge = 0.0
        battery_discharge = 0.0
        grid_import = 0.0
        unmet_demand = 0.0

        if action == self.CHARGE and surplus > 0:
            available_capacity = (
                self.BATTERY_CAPACITY
                - battery_energy
            )

            battery_charge = min(
                surplus,
                self.MAX_BATTERY_POWER,
                available_capacity,
            )

            battery_energy += (
                battery_charge
                * self.BATTERY_EFFICIENCY
            )

        if action == self.DISCHARGE and deficit > 0:
            battery_discharge = min(
                deficit,
                self.MAX_BATTERY_POWER,
                battery_energy,
            )

            battery_energy -= battery_discharge

            deficit -= (
                battery_discharge
                * self.BATTERY_EFFICIENCY
            )

        if deficit > 0:
            if grid_available == 1:
                grid_import = deficit
            else:
                unmet_demand = deficit

        self.soc = (
            battery_energy
            / self.BATTERY_CAPACITY
        )

        self.soc = float(
            np.clip(
                self.soc,
                0.0,
                1.0,
            )
        )

        reward = (
            solar_used
            - grid_import
            - (5.0 * unmet_demand)
            - (0.01 * battery_discharge)
        )

        self.current_step += 1

        terminated = (
            self.current_step
            >= len(self.data)
        )

        truncated = False

        if terminated:
            next_state = np.zeros(
                self.observation_space.shape,
                dtype=np.float32,
            )
        else:
            next_state = self._get_state()

        info = {
            "pv_production": pv,
            "consumption": consumption,
            "soc": self.soc,
            "grid_available": grid_available,
            "battery_charge": battery_charge,
            "battery_discharge": battery_discharge,
            "grid_import": grid_import,
            "unmet_demand": unmet_demand,
            "solar_used": solar_used,
        }

        return (
            next_state,
            float(reward),
            terminated,
            truncated,
            info,
        )

    def render(self):
        """Aucun rendu graphique natif n'est nécessaire."""

        return None

    def close(self):
        """Libère les éventuelles ressources de l'environnement."""

        return None