import numpy as np
from pyheatmy.config import *

from pyheatmy.params import Param


def check_IntrinK(value):
    """Raise a ValueError unless value is an intrinsic permeability in m2, 0 < value <= INTRINK_MAX.
    A value such as 11 is the former -log10(k) notation, for 1e-11 m2."""
    if not 0 < value <= INTRINK_MAX:
        raise ValueError(
            f"IntrinK = {value} is not an intrinsic permeability in m2: expected 0 < IntrinK <= {INTRINK_MAX}. "
            "A value such as 11 is the former -log10(k) notation: write 1e-11 instead."
        )


class Layer:
    """A layer of the column, from the bottom of the layer above (or the river bed) down to zLow, with its
    physical parameters. The priors of an estimation are not here: they belong to the inversion, outside
    pyheatmy."""

    def __init__(
        self,
        name: str,
        zLow: float,  # profondeur en m du bas de la couche (>0)
        IntrinK: float = INTRINK_DEFAULT,  # m2
        n: float = N_DEFAULT,
        lambda_s: float = LAMBDA_S_DEFAULT,
        rhos_cs: float = RHOS_CS_DEFAULT,
        q_s: float = Q_S_DEFAULT,
    ):
        # IntrinK est en m2 : les valeurs de l'ancienne notation -log10(k) sont refusées.
        check_IntrinK(IntrinK)

        self.name = name
        self.zLow = zLow
        self.params = Param(IntrinK, n, lambda_s, rhos_cs, q_s)

    def __repr__(self) -> str:
        return self.name + f" : ends at {self.zLow} m. " + self.params.__repr__()

    @classmethod
    def from_dict(cls, monolayer_dict):
        return cls(**monolayer_dict)

    def get_physical_params(self):
        """Return the physical parameters of the layer (Param)."""
        return self.params


def getListParameters(layersList, z_cells):
    """
    Renvoie les profils des paramètres physiques sur les cellules de la colonne.
    z_cells : profondeurs des centres des cellules (Column._z_solve), mesurées depuis le lit de la rivière.
    Une cellule appartient à la couche telle que zLow_prev < z <= zLow ; la dernière couche
    couvre toutes les cellules restantes (prolongée ou coupée au fond de la colonne).
    """
    listParameters = np.zeros((len(z_cells), len(Param._fields)))
    zLow_prev = 0
    for i, layer in enumerate(layersList):
        if i == len(layersList) - 1:
            mask = z_cells > zLow_prev
        else:
            mask = (z_cells > zLow_prev) & (z_cells <= layer.zLow)
        listParameters[mask, :] = layer.get_physical_params()
        zLow_prev = layer.zLow
    return tuple(listParameters[:, i] for i in range(len(Param._fields)))


# Add-on function to create the list of layers.
def layersListCreator(layers_spec):
    """
    Construct and return a list of Layer instances.

    Accepts:
    - layers_spec: iterable of tuples or dicts describing each layer.
      If an element is a dict, Layer.from_dict is used.
      If an element is a sequence, it is passed to Layer(*sequence).

    Expected tuple order (compatible with existing calls):
      (name, zLow, IntrinK, n, lambda_s, rhos_cs[, q_s])

    Returns:
      list of Layer
    """
    layers = []
    for spec in layers_spec:
        if isinstance(spec, dict):
            layer = Layer.from_dict(spec)
        else:
            # allow lists/tuples with the exact arguments for Layer.__init__
            layer = Layer(*spec)
        layers.append(layer)
    return layers
