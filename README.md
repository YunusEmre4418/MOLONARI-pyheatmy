# pyheatmy

Direct model of heat and water transport through a 1D riverbed sediment column, part of the MOLONARI project.

`pyheatmy` is built around the `Column` class (`pyheatmy/core.py`): a column of sediment layers between the river
and the aquifer, with the pressure and temperature measurements of a MOLONARI device. It computes the transient
head and temperature profiles and estimates the layer parameters with an MCMC, whose generic tools (priors,
DREAM proposal, energy, Gelman-Rubin diagnostic) are in
[MOLONARI-bayesian-inversion](https://github.com/YunusEmre4418/MOLONARI-bayesian-inversion).

## Installation

Install `bayesian_inversion` first, then `pyheatmy`, from the two checkouts:

```sh
pip install -e path/to/bayesian_inversion
pip install -e path/to/pyheatmy
```

## Tests

```sh
python -m pytest tests            # about 2 minutes
python -m pytest tests -m "not slow"
```

## Tutorials

The tutorials (generation of synthetic data, inversion) are in
[MOLONARI-tuto](https://github.com/YunusEmre4418/MOLONARI-tuto).
