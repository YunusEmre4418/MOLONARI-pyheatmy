"""Integration test of the direct model against an analytical solution: the steady Darcy flux through
two layers in series, with constant boundary conditions."""

import datetime

import numpy as np
import pytest

from pyheatmy import Column, Layer
from pyheatmy.params import calc_K

N_TIMES = 400                                     # 100 h of 15 min steps: long enough to reach the steady state
TIME_STEP = datetime.timedelta(minutes=15)
START = datetime.datetime(2026, 1, 1)
RIVER_HEAD = 0.05                                 # m, head difference between the river and the aquifer
RIVER_TEMPERATURE = 290.0                         # K
SENSOR_TEMPERATURES = np.array([288.0, 287.0, 286.0, 285.0])
DEPTH_SENSORS = [0.1, 0.2, 0.3, 0.4]
COLUMN_HEIGHT = DEPTH_SENSORS[-1]
N_CELLS = 40

UPPER_INTRINK = 1e-12                             # m2
LOWER_INTRINK = 1e-13
POROSITY = 0.1
LAMBDA_S = 2.0
RHOS_CS = 4e6

FLUX_RTOL = 1e-3

def steady_column(interface_depth, tmp_path):
    times = [START + i * TIME_STEP for i in range(N_TIMES)]
    column = Column.from_dict({
        "river_bed": 1.0,
        "depth_sensors": DEPTH_SENSORS,
        "offset": 0.0,
        "dH_measures": [(t, (RIVER_HEAD, RIVER_TEMPERATURE)) for t in times],
        "T_measures": [(t, SENSOR_TEMPERATURES) for t in times],
        "sigma_meas_P": None,
        "sigma_meas_T": None,
        "inter_mode": "linear",
        "nb_cells": N_CELLS,
        "rac": str(tmp_path),
    })
    column.set_layers([
        Layer.from_dict({"name": "upper", "zLow": interface_depth, "IntrinK": UPPER_INTRINK, "n": POROSITY,
                         "lambda_s": LAMBDA_S, "rhos_cs": RHOS_CS, "q_s": 0.0}),
        Layer.from_dict({"name": "lower", "zLow": COLUMN_HEIGHT, "IntrinK": LOWER_INTRINK, "n": POROSITY,
                         "lambda_s": LAMBDA_S, "rhos_cs": RHOS_CS, "q_s": 0.0}),
    ])
    column.compute_solve_transi(verbose=False)
    return column


@pytest.mark.parametrize("interface_depth", [
    pytest.param(0.2, id="interface_between_cell_centres"),
    pytest.param(0.205, id="interface_on_a_cell_centre"),
])
def test_steady_flux_through_two_layers_is_head_over_series_resistance(interface_depth, tmp_path):
    """At steady state, the Darcy flux is the same in every cell and equals
    delta H / (L1 / K1 + L2 / K2), with K = rho g k / mu (mass conservation, layers in series)."""
    column = steady_column(interface_depth, tmp_path)
    upper_K, lower_K = calc_K(np.array([UPPER_INTRINK, LOWER_INTRINK]))
    resistance = interface_depth / upper_K + (COLUMN_HEIGHT - interface_depth) / lower_K
    expected_flux = RIVER_HEAD / resistance

    final_flux = np.asarray(column.get_flows_solve())[:, -1]
    np.testing.assert_allclose(final_flux, expected_flux, rtol=FLUX_RTOL)
