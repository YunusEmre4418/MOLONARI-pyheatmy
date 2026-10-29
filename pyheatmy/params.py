from collections import namedtuple
from pyheatmy.config import *
try:
    from bayesian_inversion import Prior  # moved to bayesian_inversion, still importable from pyheatmy.params
except ImportError as error:  # bayesian_inversion is optional: the direct model runs without it
    INVERSION_IMPORT_ERROR = error
else:
    INVERSION_IMPORT_ERROR = None


PARAM_LIST = ["IntrinK", "n", "lambda_s", "rhos_cs", "q_s"]

Param = namedtuple("Param", PARAM_LIST)


def calc_K(intrinsic_permeability, viscosity=MU_W):
    """Hydraulic conductivity K (m/s) of a medium of intrinsic permeability k (m2), for water of dynamic
    viscosity mu (Pa.s): K = rho_w g k / mu. Every computation of K from k in pyheatmy goes through here."""
    return RHO_W * G * intrinsic_permeability / viscosity


def require_inversion(feature):
    """Raise an ImportError naming feature when bayesian_inversion, which the priors and the MCMC need,
    is not installed."""
    if INVERSION_IMPORT_ERROR is not None:
        raise ImportError(
            f"{feature} needs bayesian_inversion, the optional dependency 'inversion' of pyheatmy"
        ) from INVERSION_IMPORT_ERROR
