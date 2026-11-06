"""Unit tests of Column.simulate, the direct model as a function of the layer parameters, and of
Column.get_measured_temperatures: simulate gives to the bit what compute_solve_transi gives with the same layers,
leaves the column unchanged and keeps the values of the layer for the names it is not given."""

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
TOP = {"name": "silt", "zLow": 0.2, "IntrinK": 1e-13, "n": 0.2, "lambda_s": 1.5, "rhos_cs": 3e6, "q_s": 0.0}
BOTTOM = {"name": "sand", "zLow": DEPTH_SENSORS[-1], "IntrinK": 1e-12, "n": 0.1,
          "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 0.0}
OTHER = [{"IntrinK": 5e-12, "lambda_s": 3.0}, {"IntrinK": 2e-11, "n": 0.15}]
N_CELLS = 10
N_INNER_SENSORS = len(DEPTH_SENSORS) - 1          # the deepest sensor is the bottom boundary condition


@pytest.fixture
def column(tmp_path):
    series = synthetic_MOLONARI.from_dict(SERIES)
    column = Column(river_bed=1.0, depth_sensors=DEPTH_SENSORS, offset=0.0, dH_measures=series._molonariP_data,
                    T_measures=series._molonariT_data, nb_cells=N_CELLS, rac=str(tmp_path))
    column.set_layers([Layer.from_dict(TOP), Layer.from_dict(BOTTOM)])
    return column


def solved_with(column, layers_params):
    """compute_solve_transi on the column with the layers changed as simulate changes them."""
    column.set_layers([Layer.from_dict({**spec, **params}) for spec, params in zip((TOP, BOTTOM), layers_params)])
    column.compute_solve_transi(verbose=False)
    return column


def test_simulate_is_the_direct_model_to_the_bit(column):
    output = column.simulate(OTHER)
    solved = solved_with(column, OTHER)
    assert np.array_equal(output["temperatures"], solved.get_temperatures_solve())
    assert np.array_equal(output["flows"], solved.get_flows_solve())
    assert np.array_equal(output["sensors"], solved.get_temperatures_solve()[solved.get_id_sensors()])


def test_simulate_leaves_the_column_unchanged(column):
    column.compute_solve_transi(verbose=False)
    params = [layer.get_physical_params() for layer in column.all_layers]
    temperatures = column.get_temperatures_solve().copy()
    column.simulate(OTHER)
    assert [layer.get_physical_params() for layer in column.all_layers] == params
    assert np.array_equal(column.get_temperatures_solve(), temperatures)


def test_names_not_given_keep_the_values_of_the_layers(column):
    output = column.simulate([{}, {}])
    column.compute_solve_transi(verbose=False)
    assert np.array_equal(output["temperatures"], column.get_temperatures_solve())


def test_simulate_needs_the_parameters_of_every_layer(column):
    with pytest.raises(ValueError, match="2 layers"):
        column.simulate(OTHER[:1])


def test_measured_temperatures_are_the_inner_sensors(column):
    measured = column.get_measured_temperatures()
    output = column.simulate([{}, {}])
    assert measured.shape == output["sensors"].shape
    assert measured.shape[0] == N_INNER_SENSORS
