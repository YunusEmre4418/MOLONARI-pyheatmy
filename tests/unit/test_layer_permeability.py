"""Unit tests of the check on the intrinsic permeability of a Layer (pyheatmy.layers): IntrinK is in m2,
so a value, a prior bound or a fixed prior value outside 0 < IntrinK <= INTRINK_MAX is refused, in
particular the values of the former -log10(k) notation (11 for 1e-11 m2)."""

import math

import pytest

from pyheatmy import Layer
from pyheatmy.config import INTRINK_DEFAULT, INTRINK_MAX
from pyheatmy.layers import layersListCreator
from pyheatmy.params import Prior

NAME = "L"
Z_LOW = 0.4
SIGMA = 0.01
JUST_ABOVE = 1.01

OLD_NOTATION = 11                       # -log10(k) of 1e-11 m2
OLD_NOTATION_RANGE = (4 + 7, 9 + 7)     # prior range of demoPyheatmyMultilayer, in -log10(k)
INVALID_INTRINK = [OLD_NOTATION, 0.0, -1e-12, INTRINK_MAX * JUST_ABOVE, math.nan]

# m2 values of the demo notebooks and of the tests, and the bounds of the accepted range
VALID_INTRINK = [1e-12, INTRINK_DEFAULT, 1e-18, INTRINK_MAX]
VALID_INTRINK_PRIORS = [((1e-18, 1e-15), 5e-17), ((1e-16, 1e-11), 7e-16), (1e-12, SIGMA)]
INVALID_INTRINK_PRIORS = [
    (OLD_NOTATION_RANGE, SIGMA),
    ((1e-15, OLD_NOTATION), SIGMA),     # only the upper bound is wrong
    ((0.0, 1e-11), SIGMA),
    (OLD_NOTATION, SIGMA),              # fixed value
]

OTHER_PRIORS = {
    "Prior_n": ((0.01, 0.25), 0.0125),
    "Prior_lambda_s": ((1, 10), 0.5),
    "Prior_rhos_cs": ((1e6, 1e7), 9e5),
    "Prior_q_s": ((-1e-6, 1e-6), 1e-11),
}


def priors_with(prior_intrinK):
    return {"Prior_IntrinK": prior_intrinK, **OTHER_PRIORS}


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


@pytest.mark.parametrize("prior", INVALID_INTRINK_PRIORS)
def test_layer_refuses_prior_outside_range(prior):
    with pytest.raises(ValueError, match="IntrinK"):
        Layer(NAME, Z_LOW, Prior_IntrinK=Prior(*prior))


@pytest.mark.parametrize("prior", INVALID_INTRINK_PRIORS)
def test_set_priors_refuses_prior_outside_range(prior):
    with pytest.raises(ValueError, match="IntrinK"):
        Layer(NAME, Z_LOW).set_priors_from_dict(priors_with(prior))


def test_refused_priors_leave_layer_unchanged():
    layer = Layer(NAME, Z_LOW)
    layer.set_priors_from_dict(priors_with(VALID_INTRINK_PRIORS[0]))
    priors_before, params_before = list(layer.Prior_list), layer.mcmc_params
    with pytest.raises(ValueError):
        layer.set_priors_from_dict(priors_with(INVALID_INTRINK_PRIORS[0]))
    assert layer.Prior_list == priors_before
    assert layer.mcmc_params == params_before


@pytest.mark.parametrize("prior", VALID_INTRINK_PRIORS)
def test_set_priors_accepts_prior_in_m2(prior):
    layer = Layer(NAME, Z_LOW)
    layer.set_priors_from_dict(priors_with(prior))
    assert layer.Prior_IntrinK.user_range == prior[0]
