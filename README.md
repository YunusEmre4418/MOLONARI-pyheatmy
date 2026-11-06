# pyheatmy

Direct model of heat and water transport through a 1D riverbed sediment column, part of the MOLONARI project.

`pyheatmy` is built around the `Column` class (`pyheatmy/core.py`): a column of sediment layers between the river
and the aquifer, with the pressure and temperature measurements of a MOLONARI device. It computes the transient
head and temperature profiles. A `Layer` holds the physical parameters of a layer (`IntrinK` in m2, `n`,
`lambda_s`, `rhos_cs`, `q_s`).

`Column.simulate` is the direct model as a function of the layer parameters: it takes one mapping of parameter
names to values per layer, leaves the column unchanged and returns the temperatures at the inner sensors
(`"sensors"`, comparable with `Column.get_measured_temperatures()`) and the temperatures and specific flows of
every cell (`"temperatures"`, `"flows"`). An estimation of the parameters uses it as its model: the MCMC of
[MOLONARI-bayesian-inversion](https://github.com/YunusEmre4418/MOLONARI-bayesian-inversion) does, without
`pyheatmy` and `bayesian_inversion` importing each other.

## Installation

```sh
pip install -e path/to/pyheatmy
```

## Tests

```sh
python -m pytest tests
```

The tests that use `pyheatmy` together with `bayesian_inversion` are in the MOLONARI-computo tree
(`integration_tests/`, `pixi run test-all`).

## Tutorials

The tutorials (generation of synthetic data, inversion) are in
[MOLONARI-tuto](https://github.com/YunusEmre4418/MOLONARI-tuto).
