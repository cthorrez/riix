"""mathematical constants computed once here to avoid recomputation"""
import math

# general math constants
PI2 = math.pi**2.0
THREE_OVER_PI_SQUARED = 3.0 / PI2

# glicko constants
Q = math.log(10.0) / 400.0
Q2 = Q**2.0
Q2_3 = 3.0 * Q2
Q2_3_OVER_PI2 = Q2_3 / PI2
