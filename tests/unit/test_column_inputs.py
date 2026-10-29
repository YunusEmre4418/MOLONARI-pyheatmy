"""Unit tests of the input checks of Column: inconsistent measurements are refused with a ValueError that says
what is wrong, before any computation. The river and aquifer temperatures and the head differential are boundary
conditions, needed at every time; the temperatures of the sensors are only compared with the model, so missing
values (NaN) there are accepted (see test_missing_measurements.py)."""

import datetime

import numpy as np
import pytest

from pyheatmy import Column
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
SHIFT = datetime.timedelta(hours=6)
I_TIME = 10                                       # a time inside the series
LAST = -1                                         # aquifer temperature: last value of each T_measures item
EXTRA_DEPTH = 0.5


@pytest.fixture(scope="module")
def measures():
    series = synthetic_MOLONARI.from_dict(SERIES)
    return series._molonariP_data, series._molonariT_data


def column(dH_measures, T_measures, tmp_path, depth_sensors=DEPTH_SENSORS):
    return Column.from_dict({"river_bed": 1.0, "depth_sensors": depth_sensors, "offset": 0.0,
                             "dH_measures": dH_measures, "T_measures": T_measures, "rac": str(tmp_path)})


def with_dH(dH_measures, i, dH=None, T_riv=None):
    out = list(dH_measures)
    date, (d, t) = out[i]
    out[i] = (date, (d if dH is None else dH, t if T_riv is None else T_riv))
    return out


def with_T(T_measures, i, k, value):
    out = [(date, np.array(t, dtype=float)) for date, t in T_measures]
    out[i][1][k] = value
    return out


def with_dates(measurements, dates):
    return [(date, values) for date, (_, values) in zip(dates, measurements)]


def test_valid_measurements_are_accepted(measures, tmp_path):
    dH_measures, T_measures = measures
    column(dH_measures, T_measures, tmp_path)


def test_different_numbers_of_times_are_refused(measures, tmp_path):
    dH_measures, T_measures = measures
    with pytest.raises(ValueError, match="same number of times"):
        column(dH_measures, T_measures[:LAST], tmp_path)


def test_different_dates_are_refused(measures, tmp_path):
    dH_measures, T_measures = measures
    shifted = [(date + SHIFT, values) for date, values in T_measures]
    with pytest.raises(ValueError, match="same dates"):
        column(dH_measures, shifted, tmp_path)


@pytest.mark.parametrize("change", ["repeated", "swapped"])
def test_dates_that_do_not_increase_are_refused(measures, tmp_path, change):
    dH_measures, T_measures = measures
    dates = [date for date, _ in dH_measures]
    if change == "repeated":
        dates[I_TIME + 1] = dates[I_TIME]         # time step 0
    else:
        dates[I_TIME], dates[I_TIME + 1] = dates[I_TIME + 1], dates[I_TIME]  # negative time step
    with pytest.raises(ValueError, match="strictly increasing"):
        column(with_dates(dH_measures, dates), with_dates(T_measures, dates), tmp_path)


@pytest.mark.parametrize("depth_sensors", [DEPTH_SENSORS[:LAST], DEPTH_SENSORS + [EXTRA_DEPTH]], ids=["fewer", "more"])
def test_depths_that_do_not_match_the_temperatures_are_refused(measures, tmp_path, depth_sensors):
    dH_measures, T_measures = measures
    with pytest.raises(ValueError, match="one temperature per depth"):
        column(dH_measures, T_measures, tmp_path, depth_sensors=depth_sensors)


@pytest.mark.parametrize("boundary", ["head differential", "river temperature", "aquifer temperature"])
def test_missing_boundary_conditions_are_refused(measures, tmp_path, boundary):
    dH_measures, T_measures = measures
    if boundary == "head differential":
        dH_measures = with_dH(dH_measures, I_TIME, dH=np.nan)
    elif boundary == "river temperature":
        dH_measures = with_dH(dH_measures, I_TIME, T_riv=np.nan)
    else:
        T_measures = with_T(T_measures, I_TIME, LAST, np.nan)
    with pytest.raises(ValueError, match=f"NaN.*{boundary}"):
        column(dH_measures, T_measures, tmp_path)
