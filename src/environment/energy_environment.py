import math
import numpy as np


class EnergyEnvironment:
    """
    Solar Home Energy Environment

    State:
        [
            pv_normalized,
            consumption_normalized,
            soc,
            grid_available,
            sin_hour,
            cos_hour
        ]

    Actions:
        0 -> IDLE
        1 -> CHARGE
        2 -> DISCHARGE
    """

    IDLE = 0
    CHARGE = 1
    DISCHARGE = 2

    BATTERY_CAPACITY = 10.0
    INITIAL_SOC = 0.50
    MAX_BATTERY_POWER = 2.0
    BATTERY_EFFICIENCY = 0.95

    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.soc = self.INITIAL_SOC

        self.pv_max = max(
            float(data["pv_production"].max()),
            1.0
        )

        self.consumption_max = max(
            float(data["consumption"].max()),
            1.0
        )

    def reset(self):
        self.current_step = 0
        self.soc = self.INITIAL_SOC
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]

        pv = row["pv_production"]
        consumption = row["consumption"]
        grid_available = row["grid_available"]

        pv_normalized = min(pv / self.pv_max, 1.0)
        consumption_normalized = min(
            consumption / self.consumption_max,
            1.0
        )

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

        if action not in [
            self.IDLE,
            self.CHARGE,
            self.DISCHARGE,
        ]:
            raise ValueError(
                f"Invalid action: {action}"
            )

        row = self.data.iloc[self.current_step]

        pv = float(row["pv_production"])
        consumption = float(row["consumption"])
        grid_available = int(row["grid_available"])

        battery_energy = (
            self.soc * self.BATTERY_CAPACITY
        )

        solar_used = min(
            pv,
            consumption
        )

        surplus = max(
            pv - consumption,
            0.0
        )

        deficit = max(
            consumption - pv,
            0.0
        )

        battery_charge = 0.0
        battery_discharge = 0.0
        grid_import = 0.0
        unmet_demand = 0.0

        # CHARGE
        if action == self.CHARGE and surplus > 0:

            available_capacity = (
                self.BATTERY_CAPACITY
                - battery_energy
            )

            battery_charge = min(
                surplus,
                self.MAX_BATTERY_POWER,
                available_capacity
            )

            battery_energy += (
                battery_charge
                * self.BATTERY_EFFICIENCY
            )

        # DISCHARGE
        if action == self.DISCHARGE and deficit > 0:

            battery_discharge = min(
                deficit,
                self.MAX_BATTERY_POWER,
                battery_energy
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

        self.soc = max(
            0.0,
            min(1.0, self.soc)
        )

        reward = (
            solar_used
            - grid_import
            - (5 * unmet_demand)
            - (0.01 * battery_discharge)
        )

        self.current_step += 1

        done = (
            self.current_step
            >= len(self.data)
        )

        if done:
            next_state = np.zeros(
                6,
                dtype=np.float32
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
            reward,
            done,
            info,
        )