"""Unit tests of the check on the intrinsic permeability of a Layer (pyheatmy.layers): IntrinK is in m2,
so a value outside 0 < IntrinK <= INTRINK_MAX is refused, in particular the values of the former -log10(k)
notation (11 for 1e-11 m2). An MCMC whose prior gives such values stops at its first evaluation: that test needs
bayesian_inversion and is in the tests of the MOLONARI-computo tree."""

import math

import pytest

from pyheatmy import Layer
from pyheatmy.config import INTRINK_DEFAULT, INTRINK_MAX
from pyheatmy.layers import layersListCreator

NAME = "L"
Z_LOW = 0.4
JUST_ABOVE = 1.01

OLD_NOTATION = 11                       # -log10(k) of 1e-11 m2
INVALID_INTRINK = [OLD_NOTATION, 0.0, -1e-12, INTRINK_MAX * JUST_ABOVE, math.nan]

# m2 values of the demo notebooks and of the tests, and the bounds of the accepted range
VALID_INTRINK = [1e-12, INTRINK_DEFAULT, 1e-18, INTRINK_MAX]


@pytest.mark.parametrize("value", INVALID_INTRINK)
def test_layer_refuses_intrinK_outside_range(value):
    with pytest.raises(ValueError, match="IntrinK"):
        Layer(NAME, Z_LOW, IntrinK=value)


@pytest.mark.parametrize("value", VALID_INTRINK)
def test_layer_accepts_intrinK_in_m2(value):
    layer = Layer(NAME, Z_LOW, IntrinK=value)
    assert layer.get_physical_params().IntrinK == value


def test_error_message_gives_value_and_range():
    with pytest.raises(ValueError) as error:
        Layer(NAME, Z_LOW, IntrinK=OLD_NOTATION)
    assert str(OLD_NOTATION) in str(error.value)
    assert str(INTRINK_MAX) in str(error.value)


def test_layers_list_creator_refuses_old_notation():
    with pytest.raises(ValueError, match="IntrinK"):
        layersListCreator([(NAME, Z_LOW, OLD_NOTATION, 0.1, 2.0, 4e6)])
