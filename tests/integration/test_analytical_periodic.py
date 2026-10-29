"""Integration test of the direct model against the periodic analytical solution of Analy_Sol (pyheatmy.val_analy):
a sinusoidal river temperature over a semi-infinite medium, with a constant Darcy flux,
T(z, t) = T_mean + A exp(-a z) sin(2 pi t / P + phase - b z)."""

import numpy as np
import pytest

from pyheatmy import Analy_Sol, Column, Layer
from pyheatmy.config import NSECINDAY, ZERO_CELSIUS
from pyheatmy.synthetic_MOLONARI import synthetic_MOLONARI

START = (2026, 1, 1, 0, 0, 0)
END = (2026, 1, 21, 0, 0, 0)                      # 20 days: the initial transient (time scale L^2 / (pi^2 kappa) ~ 2.6 days) has died out
TIME_STEP = 900                                   # s
DEPTH_SENSORS = [0.1, 0.2, 0.3, 1.0]              # a 1 m column, so that the bottom boundary barely acts above 0.4 m
N_CELLS = 100
T_MEAN = ZERO_CELSIUS + 12
AMPLITUDE = 2.0                                   # K
PERIOD = NSECINDAY

LAYER = {"name": "sand", "zLow": DEPTH_SENSORS[-1], "IntrinK": 1e-12, "n": 0.1,
         "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 0.0}

COMPARED_DEPTH = 0.4                              # m, cells compared with the analytical solution
COMPARED_DURATION = 2 * PERIOD                    # the last two days
# Measured with this set-up: 0.0040 K (conduction) and 0.0035-0.0047 K (dH = +-0.1 m); the velocity with the wrong
# sign gives 0.18-0.19 K. With the former float32 temperature solver, the differences were 0.012-0.025 K.
TEMPERATURE_ATOL = 0.01                           # K
PHASE = np.pi / 3                                 # rad, a non-zero phase of the river signal


def synthetic_series(dH, phase=0.0):
    return synthetic_MOLONARI.from_dict({
        "offset": 0.0,
        "depth_sensors": DEPTH_SENSORS,
        "param_time_dates": [START, END, TIME_STEP],
        "param_dH_signal": [[dH]],
        "param_T_riv_signal": [[T_MEAN], [AMPLITUDE, PERIOD, phase]],
        "param_T_aq_signal": [[T_MEAN]],
        "sigma_meas_P": None,
        "sigma_meas_T": None,
        "verbose": False,
    })


def column_of(series, tmp_path):
    return Column.from_dict({
        "river_bed": 1.0,
        "depth_sensors": DEPTH_SENSORS,
        "offset": 0.0,
        "dH_measures": series._molonariP_data,
        "T_measures": series._molonariT_data,
        "sigma_meas_P": None,
        "sigma_meas_T": None,
        "inter_mode": "linear",
        "nb_cells": N_CELLS,
        "rac": str(tmp_path),
    })


@pytest.mark.parametrize("dH, phase", [
    pytest.param(0.0, 0.0, id="conduction"),
    pytest.param(0.1, 0.0, id="downward_flow"),
    pytest.param(-0.1, 0.0, id="upward_flow"),
    pytest.param(0.1, PHASE, id="downward_flow_with_phase"),
])
def test_direct_model_matches_periodic_analytical_solution(dH, phase, tmp_path):
    series = synthetic_series(dH, phase)
    column = column_of(series, tmp_path)
    layer = Layer.from_dict(LAYER)
    column.set_layers(layer)
    column.compute_solve_transi(verbose=False)

    analytical = Analy_Sol(column, series, layer, N_CELLS)
    analytical.compute_temp_general()

    times = np.asarray(series._time_array)
    compared_times = times >= times[-1] - COMPARED_DURATION
    compared_cells = analytical._z_solve <= COMPARED_DEPTH
    direct = np.asarray(column.get_temperatures_solve())
    np.testing.assert_allclose(direct[compared_cells][:, compared_times],
                               analytical.analy_temp_general[compared_cells][:, compared_times],
                               rtol=0, atol=TEMPERATURE_ATOL)


def test_general_solution_without_flow_is_the_conduction_solution(tmp_path):
    series = synthetic_series(0.0, PHASE)
    analytical = Analy_Sol(column_of(series, tmp_path), series, Layer.from_dict(LAYER), N_CELLS)
    analytical.compute_temp_general()
    analytical.compute_temp_cond()
    np.testing.assert_allclose(analytical.analy_temp_general, analytical.analy_temp_cond, rtol=1e-12, atol=0)


def test_rmse_is_root_mean_square_difference_at_sensors(tmp_path):
    series = synthetic_series(0.1)
    column = column_of(series, tmp_path)
    layer = Layer.from_dict(LAYER)
    analytical = Analy_Sol(column, series, layer, N_CELLS)
    rmse = analytical.generate_RMSE_analytical(series, column, layer)

    analytical_sensors = analytical.analy_temp_general[analytical._id_sensors, :]
    direct_sensors = np.asarray(column.get_temperatures_solve())[column.get_id_sensors(), :]
    expected = np.sqrt(np.mean((analytical_sensors - direct_sensors) ** 2, axis=1))
    np.testing.assert_allclose(rmse[:-1], expected, rtol=1e-6, atol=0)
    np.testing.assert_allclose(rmse[-1], np.sqrt(np.mean(expected ** 2)), rtol=1e-6, atol=0)


@pytest.mark.parametrize("dH_signal, T_riv_signal, T_aq_signal", [
    pytest.param([[0.1], [0.05, PERIOD, 0.0]], [[T_MEAN], [AMPLITUDE, PERIOD, 0.0]], [[T_MEAN]], id="varying_head"),
    pytest.param([[0.1]], [[T_MEAN], [AMPLITUDE, PERIOD, 0.0], [AMPLITUDE, 2 * PERIOD, 0.0]], [[T_MEAN]],
                 id="two_river_components"),
    pytest.param([[0.1]], [[T_MEAN], [AMPLITUDE, PERIOD, 0.0]], [[T_MEAN + 1]], id="different_aquifer_mean"),
])
def test_refuses_boundary_conditions_without_analytical_solution(dH_signal, T_riv_signal, T_aq_signal, tmp_path):
    series = synthetic_series(0.1)
    column = column_of(series, tmp_path)
    series._param_dH, series._param_T_riv, series._param_T_aq = dH_signal, T_riv_signal, T_aq_signal
    with pytest.raises(ValueError, match="Analy_Sol"):
        Analy_Sol(column, series, Layer.from_dict(LAYER), N_CELLS)


def test_refuses_source_term(tmp_path):
    series = synthetic_series(0.1)
    with pytest.raises(ValueError, match="q_s"):
        Analy_Sol(column_of(series, tmp_path), series, Layer.from_dict({**LAYER, "q_s": 1e-8}), N_CELLS)
