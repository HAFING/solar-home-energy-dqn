import math
import numpy as np


class EnergyEnvironment:

    # Actions
    IDLE = 0
    CHARGE = 1
    DISCHARGE = 2

    # Battery parameters
    BATTERY_CAPACITY = 10.0
    INITIAL_SOC = 0.50
    MAX_BATTERY_POWER = 2.0
    BATTERY_EFFICIENCY = 0.95

    def __init__(self, data):
        self.data = data
        self.current_step = 0
        self.soc = self.INITIAL_SOC

    def reset(self):
        self.current_step = 0
        self.soc = self.INITIAL_SOC

        return self._get_state()

    def step(self, action):
        pass

    def _get_state(self):
        pass