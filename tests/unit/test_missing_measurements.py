"""Unit tests of missing temperature measurements (NaN) at the sensors of a Column: they are left out of the
initial profile and of the RMSE, and the user is told once how many are missing. That they are left out of the
energy of the MCMC is tested in bayesian_inversion (tests/test_energy.py)."""

import warnings

import numpy as np
import pytest

from pyheatmy import Column, Layer
from pyheatmy.config import NSECINDAY, ZERO_CELSIUS
from pyheatmy.synthetic_MOLONARI import synthetic_MOLONARI

DEPTH_SENSORS = [0.1, 0.2, 0.3, 0.4]
T_MEAN = ZERO_CELSIUS + 12
SERIES = {
    "offset": 0.0,
    "depth_sensors": DEPTH_SENSORS,
    "param_time_dates": [(2026, 1, 1, 0, 0, 0), (2026, 1, 2, 0, 0, 0), 900],   # 1 day, 96 times
    "param_dH_signal": [[0.1]],
    "param_T_riv_signal": [[T_MEAN], [2.0, NSECINDAY, 0.0]],
    "param_T_aq_signal": [[T_MEAN]],
    "sigma_meas_P": None,
    "sigma_meas_T": None,
    "verbose": False,
}
LAYER = {"name": "sand", "zLow": DEPTH_SENSORS[-1], "IntrinK": 1e-12, "n": 0.1,
         "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 0.0}
N_CELLS = 10
FIRST = 0
MIDDLE_SENSOR = 1                                 # 0.2 m
EMPTY_SENSOR = 2                                  # 0.3 m, no measurement at all
MISSING_TIMES = [10, 11, 50]
MISSING = "missing"
RTOL = 1e-12
INTERP_RTOL = 1e-9                                # Lagrange formula against a polynomial fit


@pytest.fixture(scope="module")
def measures():
    series = synthetic_MOLONARI.from_dict(SERIES)
    return series._molonariP_data, series._molonariT_data


def with_nan(T_measures, times, sensor):
    out = [(date, np.array(t, dtype=float)) for date, t in T_measures]
    for i in times:
        out[i][1][sensor] = np.nan
    return out


def column(dH_measures, T_measures, tmp_path, inter_mode="linear"):
    col = Column.from_dict({"river_bed": 1.0, "depth_sensors": DEPTH_SENSORS, "offset": 0.0,
                            "dH_measures": dH_measures, "T_measures": T_measures, "nb_cells": N_CELLS,
                            "inter_mode": inter_mode, "rac": str(tmp_path)})
    col.set_layers(Layer.from_dict(LAYER))
    return col


def missing_warnings(record):
    return [str(w.message) for w in record if issubclass(w.category, UserWarning) and MISSING in str(w.message)]


def test_complete_measurements_give_no_warning(measures, tmp_path):
    dH_measures, T_measures = measures
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        column(dH_measures, T_measures, tmp_path)
    assert not missing_warnings(record)


def test_missing_measurements_are_counted_in_one_warning(measures, tmp_path):
    dH_measures, T_measures = measures
    gaps = with_nan(with_nan(T_measures, MISSING_TIMES, MIDDLE_SENSOR), range(len(T_measures)), EMPTY_SENSOR)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        column(dH_measures, gaps, tmp_path)
    messages = missing_warnings(record)
    assert len(messages) == 1
    n_sensor_values = (len(DEPTH_SENSORS) - 1) * len(T_measures)
    assert f"{len(MISSING_TIMES) + len(T_measures)} of {n_sensor_values}" in messages[0]
    assert f"{DEPTH_SENSORS[EMPTY_SENSOR]} m" in messages[0]  # the sensor without any measurement is named


@pytest.mark.parametrize("inter_mode", ["linear", "lagrange"])
def test_initial_profile_uses_the_measured_sensors(measures, tmp_path, inter_mode):
    dH_measures, T_measures = measures
    gaps = with_nan(T_measures, [FIRST], MIDDLE_SENSOR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        col = column(dH_measures, gaps, tmp_path, inter_mode=inter_mode)
    col.compute_solve_transi(verbose=False)
    first = np.array([dH_measures[FIRST][1][1], *gaps[FIRST][1]])   # river, sensors, aquifer at the first time
    z = np.array([0.0, *DEPTH_SENSORS])
    kept = ~np.isnan(first)
    z_cells = np.asarray(col.get_depths_solve())
    if inter_mode == "linear":
        expected = np.interp(z_cells, z[kept], first[kept])
    else:                                          # the polynomial through the measured points
        expected = np.polynomial.Polynomial.fit(z[kept], first[kept], deg=kept.sum() - 1)(z_cells)
    initial = np.asarray(col.get_temperatures_solve())[:, FIRST]
    assert not np.isnan(np.asarray(col.get_temperatures_solve())).any()
    assert initial == pytest.approx(expected, rel=INTERP_RTOL)


def test_rmse_is_over_the_measured_times(measures, tmp_path):
    dH_measures, T_measures = measures
    gaps = with_nan(with_nan(T_measures, MISSING_TIMES, MIDDLE_SENSOR), range(len(T_measures)), EMPTY_SENSOR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        col = column(dH_measures, gaps, tmp_path)
    col.compute_solve_transi(verbose=False)
    model = np.asarray(col.get_temperatures_solve())[col.get_id_sensors(), :]
    observed = np.array([t[:-1] for _, t in gaps]).T
    expected = []
    for sensor in range(observed.shape[0]):
        squares = [(m - o) ** 2 for m, o in zip(model[sensor], observed[sensor]) if not np.isnan(o)]
        expected.append(np.sqrt(sum(squares) / len(squares)) if squares else np.nan)
    measured_sensors = [r for r in expected if not np.isnan(r)]
    expected.append(np.sqrt(sum(r ** 2 for r in measured_sensors) / len(measured_sensors)))
    assert col.get_RMSE() == pytest.approx(expected, rel=RTOL, nan_ok=True)
