"""Unit tests of the measurement noise of synthetic_MOLONARI (pyheatmy.synthetic_MOLONARI): the noise is added to
copies of the series, so the noise-free series, the boundary conditions of a column built from them and the
temperatures this column computes stay free of noise."""

import numpy as np
import pytest

from pyheatmy import Column, Layer
from pyheatmy.config import NSECINDAY, ZERO_CELSIUS
from pyheatmy.synthetic_MOLONARI import synthetic_MOLONARI

SEED = 0
DEPTH_SENSORS = [0.1, 0.2, 0.3, 0.4]
T_MEAN = ZERO_CELSIUS + 12
SIGMA_T = 0.1                                     # K
SIGMA_P = 0.01                                    # m
SERIES = {
    "offset": 0.0,
    "depth_sensors": DEPTH_SENSORS,
    "param_time_dates": [(2026, 1, 1, 0, 0, 0), (2026, 1, 11, 0, 0, 0), 900],   # 10 days, 960 times
    "param_dH_signal": [[0.1]],
    "param_T_riv_signal": [[T_MEAN], [2.0, NSECINDAY, 0.0]],
    "param_T_aq_signal": [[T_MEAN]],
    "verbose": False,
}
N_TIMES = 960
NOISE_RTOL = 0.1                                  # the standard deviation of 960 draws is within about 2 % of sigma

LAYER = {"name": "sand", "zLow": DEPTH_SENSORS[-1], "IntrinK": 1e-12, "n": 0.1,
         "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 0.0}
N_CELLS = 20


def synthetic_series(sigma_T, sigma_P, **arguments):
    np.random.seed(SEED)
    return synthetic_MOLONARI.from_dict({**SERIES, "sigma_meas_T": sigma_T, "sigma_meas_P": sigma_P, **arguments})


def shaft(measures):
    return np.array([temperatures for _, temperatures in measures])


def generating_column(series, tmp_path):
    """Column built as in demo_genData: river from the pressure sensor, sensors and aquifer from the shaft."""
    return Column.from_dict({
        "river_bed": 1.0,
        "depth_sensors": DEPTH_SENSORS,
        "offset": 0.0,
        "dH_measures": series._molonariP_data,
        "T_measures": series._T_Shaft_measures,
        "nb_cells": N_CELLS,
        "rac": str(tmp_path),
    })


def test_noise_keeps_the_noise_free_series():
    clean = synthetic_series(None, None)
    noisy = synthetic_series(SIGMA_T, SIGMA_P)
    assert np.array_equal(noisy._T_riv, clean._T_riv)
    assert np.array_equal(noisy._dH, clean._dH)
    assert np.array_equal(noisy._T_Shaft, clean._T_Shaft)
    assert np.array_equal(shaft(noisy._T_Shaft_measures), shaft(clean._T_Shaft_measures))


def test_noise_has_the_measurement_standard_deviation():
    noisy = synthetic_series(SIGMA_T, SIGMA_P)
    assert len(noisy._dates) == N_TIMES
    assert np.std(noisy._T_riv_perturb - noisy._T_riv) == pytest.approx(SIGMA_T, rel=NOISE_RTOL)
    assert np.std(noisy._dH_perturb - noisy._dH) == pytest.approx(SIGMA_P, rel=NOISE_RTOL)
    assert np.std(shaft(noisy._molonariT_data) - noisy._T_Shaft) == pytest.approx(SIGMA_T, rel=NOISE_RTOL)


@pytest.mark.parametrize("sigma_T, sigma_P", [
    pytest.param(None, None, id="without_noise"),
    pytest.param(SIGMA_T, SIGMA_P, id="with_noise"),
])
def test_perturbed_series_are_copies(sigma_T, sigma_P):
    series = synthetic_series(sigma_T, sigma_P)
    assert not np.shares_memory(series._T_riv_perturb, series._T_riv)
    assert not np.shares_memory(series._dH_perturb, series._dH)
    assert not np.shares_memory(series._T_Shaft_perturb, series._T_Shaft)


def test_generating_column_has_a_noise_free_aquifer_temperature(tmp_path):
    noisy = synthetic_series(SIGMA_T, SIGMA_P)
    column = generating_column(noisy, tmp_path)
    assert np.array_equal(column._T_aq, noisy._T_aq)


def test_measured_column_keeps_its_computed_temperatures(tmp_path):
    noisy = synthetic_series(SIGMA_T, SIGMA_P)
    column = generating_column(noisy, tmp_path)
    column.set_layers(Layer.from_dict(LAYER))
    noisy._measures_column_one_layer(column)
    computed = np.asarray(column.get_temperature_at_sensors(verbose=False))   # river, sensors, aquifer
    assert np.array_equal(noisy._T_Shaft, computed[1:].T)
    assert np.std(shaft(noisy._molonariT_data) - noisy._T_Shaft) == pytest.approx(SIGMA_T, rel=NOISE_RTOL)


def test_given_aquifer_temperature_gives_a_noise_free_shaft():
    aquifer = np.full(N_TIMES, T_MEAN)
    clean = synthetic_series(None, None, array_T_bottom=aquifer)
    noisy = synthetic_series(SIGMA_T, SIGMA_P, array_T_bottom=aquifer)
    assert np.array_equal(shaft(noisy._T_Shaft_measures), shaft(clean._T_Shaft_measures))
