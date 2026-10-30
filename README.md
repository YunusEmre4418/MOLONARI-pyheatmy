# pyheatmy

Direct model of heat and water transport through a 1D riverbed sediment column, part of the MOLONARI project.

`pyheatmy` is built around the `Column` class (`pyheatmy/core.py`): a column of sediment layers between the river
and the aquifer, with the pressure and temperature measurements of a MOLONARI device. It computes the transient
head and temperature profiles. The estimation of the layer parameters by MCMC is in
[MOLONARI-bayesian-inversion](https://github.com/YunusEmre4418/MOLONARI-bayesian-inversion): its
`InversionColumn` is a `Column` with the MCMC.

## Installation

The direct model needs `pyheatmy` only:

```sh
pip install -e path/to/pyheatmy
```

The priors of the layers (`Layer.set_priors_from_dict`) also need `bayesian_inversion`, the optional dependency
`inversion` of `pyheatmy`. It is not on PyPI: install it from its checkout, after `pyheatmy`, on which it depends.

```sh
pip install -e path/to/pyheatmy
pip install -e path/to/bayesian_inversion
```

## Tests

```sh
python -m pytest tests
```

Without `bayesian_inversion`, the tests of the priors are skipped. The tests of the MCMC are in
`bayesian_inversion`.

## Tutorials

The tutorials (generation of synthetic data, inversion) are in
[MOLONARI-tuto](https://github.com/YunusEmre4418/MOLONARI-tuto).
