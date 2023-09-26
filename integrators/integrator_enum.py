from enum import Enum


class IntegratorEnum(Enum):
    VERLET = "Verlet"
    EULER = "Euler"
    RK4 = "RK4"
