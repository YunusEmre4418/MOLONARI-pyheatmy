import numpy as np
from numba import njit
from scipy.interpolate import interp1d, lagrange, BarycentricInterpolator


"""implémentation numba de l'interpolation de Lagrange Barycentrique, 
    voir https://people.maths.ox.ac.uk/trefethen/barycentric.pdf, formule 4.2"""


@njit
def n_barycentric(x):
    """calcul les poids barycentriques w_i du polynome interpolateur de Lagrange (qui dépendent seulement des abscisses interpollées """
    n = len(x)
    diff = np.array([[x[i]-x[j] for j in range(n)] for i in range(n)])
    for i in range(n):
        diff[i, i] = 1
    w = np.array([1/np.prod(diff[i, :]) for i in range(n)])
    return w

# @guvectorize(["void(float64[:], float64[:], float64[:], float64[:])"],
#             "(n),(n),(n),(m)->(m)")


@njit
def n_evaluate(x, w, y, point):
    """évalue le polynome interpollateur au point P, les poids w_i précedemment calculés sont nécessaires"""
    poids = w/(point-x)
    return np.dot(poids, y)/np.sum(poids)


class Lagrange():
    """encapsule ces fonctions dans un objet Lagrange similaire aux interpollateurs scipy
    Pour plus de lisibilité (similarité aux interpollateurs scipy) les ordonées de la fonction à interpoller sont donnés dès l'initialisation même si elles sont inutiles"""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.n = len(x)
        self.w = n_barycentric(x)

    def __call__(self, point):
        index = np.where(self.x == point)
        if len(index[0]) == 0:
            return n_evaluate(self.x, self.w, self.y, point)
        else:
            return self.y[index[0][0]]
