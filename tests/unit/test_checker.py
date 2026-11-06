"""Unit tests of the checker decorator (pyheatmy.checker) on Column.compute_solve_transi: the getters that need it
raise before it, and a solved column can be pickled and copied, each copy solving itself."""

import copy
import pickle

import numpy as np
import pytest

from pyheatmy import Column, Layer
from pyheatmy.checker import ComputationOrderException
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
OTHER_INTRINK = 1e-11                             # a different flow, hence different temperatures
N_CELLS = 10


@pytest.fixture
def column(tmp_path):
    series = synthetic_MOLONARI.from_dict(SERIES)
    column = Column(river_bed=1.0, depth_sensors=DEPTH_SENSORS, offset=0.0, dH_measures=series._molonariP_data,
                    T_measures=series._molonariT_data, nb_cells=N_CELLS, rac=str(tmp_path))
    column.set_layers(Layer.from_dict(LAYER))
    return column


def test_getters_need_the_computation(column):
    with pytest.raises(ComputationOrderException):
        column.get_temperatures_solve()
    column.compute_solve_transi(verbose=False)
    assert column.get_temperatures_solve().shape[0] == N_CELLS


def test_solved_column_can_be_pickled(column):
    column.compute_solve_transi(verbose=False)
    restored = pickle.loads(pickle.dumps(column))
    assert np.array_equal(restored.get_temperatures_solve(), column.get_temperatures_solve())


def test_copy_of_a_solved_column_solves_itself(column):
    column.compute_solve_transi(verbose=False)
    before = column.get_temperatures_solve().copy()
    other = copy.deepcopy(column)
    other.set_layers(Layer.from_dict({**LAYER, "IntrinK": OTHER_INTRINK}))
    other.compute_solve_transi(verbose=False)
    assert np.array_equal(column.get_temperatures_solve(), before)
    assert not np.array_equal(other.get_temperatures_solve(), before)
