"""Unit tests of Layer.set_priors_from_dict (pyheatmy.layers): assigning priors changes the MCMC space in
which the parameters are stored, not their physical values."""

import pytest

from pyheatmy import Layer

RELATIVE_TOLERANCE = 1e-9  # the log and symlog transforms round-trip to rounding error

PHYSICAL = {"IntrinK": 1e-12, "n": 0.1, "lambda_s": 2.0, "rhos_cs": 4e6, "q_s": 3e-7}
DEMO_PRIORS = {
    "Prior_IntrinK": ((1e-18, 1e-15), 5e-17),
    "Prior_n": ((0.01, 0.25), 0.0125),
    "Prior_lambda_s": ((1, 10), 0.5),
    "Prior_rhos_cs": ((1e6, 1e7), 9e5),
    "Prior_q_s": ((-1e-6, 1e-6), 1e-11),
}
OTHER_PRIORS = {**DEMO_PRIORS, "Prior_IntrinK": ((1e-16, 1e-11), 7e-16), "Prior_q_s": ((-1e-5, 1e-5), 1e-10)}


def demo_layer():
    return Layer.from_dict({"name": "L", "zLow": 0.4, **PHYSICAL})


def assert_physical_values_kept(layer):
    kept = layer.get_physical_params()._asdict()
    assert kept == pytest.approx(PHYSICAL, rel=RELATIVE_TOLERANCE, abs=0)


def test_assigning_priors_keeps_physical_values():
    layer = demo_layer()
    layer.set_priors_from_dict(DEMO_PRIORS)
    assert_physical_values_kept(layer)


def test_replacing_priors_keeps_physical_values():
    layer = demo_layer()
    layer.set_priors_from_dict(DEMO_PRIORS)
    layer.set_priors_from_dict(OTHER_PRIORS)
    assert_physical_values_kept(layer)


def test_priors_on_default_layer():
    # The default q_s is the integer 0; its symlog transform must not raise.
    layer = Layer("L", 0.4)
    layer.set_priors_from_dict(DEMO_PRIORS)
    assert layer.get_physical_params().q_s == 0
