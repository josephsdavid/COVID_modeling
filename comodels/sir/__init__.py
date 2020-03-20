"""
comodels.sir

SIR based disease models of covid 19. Inculded so far:
    Penn: Penn model (SIR)
    PennDeath: Penn model with death
"""
from .penn import Penn, calc_hosp_numbers, calc_admission_deltas, rolling_sum, PennDeath
from .sir import SIR, SIRD
