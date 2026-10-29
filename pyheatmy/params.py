from collections import namedtuple
from pyheatmy.config import *
from bayesian_inversion import Prior  # moved to bayesian_inversion, still importable from pyheatmy.params


PARAM_LIST = ["IntrinK", "n", "lambda_s", "rhos_cs", "q_s"]

Param = namedtuple("Param", PARAM_LIST)


def calc_K(intrinsic_permeability, viscosity=MU_W):
    """Hydraulic conductivity K (m/s) of a medium of intrinsic permeability k (m2), for water of dynamic
    viscosity mu (Pa.s): K = rho_w g k / mu. Every computation of K from k in pyheatmy goes through here."""
    return RHO_W * G * intrinsic_permeability / viscosity
