from typing import List, Sequence, Union
import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.colors import LinearSegmentedColormap

from pyheatmy.lagrange import Lagrange
from pyheatmy.params import calc_K
from pyheatmy.checker import checker
from pyheatmy.config import *
from pyheatmy.linear_system import *

from pyheatmy.utils import *
from pyheatmy.layers import Layer, getListParameters


# Column is a monolithic class and pyheatmy is executable from there. Calculation, retrieval and plots are methods from the column class
class Column:  # colonne de sédiments verticale entre le lit de la rivière et l'aquifère
    def __init__(
        self,
        river_bed: float,  # profondeur de la colonne en mètres
        # profondeur des capteurs de températures en mètres
        depth_sensors: Sequence[float],
        offset: float,  # correspond au décalage du capteur de température par rapport au lit de la rivière
        # liste contenant un tuple avec la date, la charge et la température au sommet de la colonne
        dH_measures: list,
        T_measures: list,  # liste contenant un tuple avec la date et la température aux points de mesure de longueur le nombre de temps mesuré
        sigma_meas_P: float = None,  # écart type de l'incertitude sur les valeurs de pression capteur
        sigma_meas_T: float = None,  # écart type de l'incertitude sur les valeurs de température capteur
        all_layers: list[Layer] = None,  # liste des couches de la colonne (ou plus tard avec set_layers)
        inter_mode: str = "linear",  # mode d'interpolation du profil de température initial : 'lagrange' ou 'linear'
        eps=EPSILON,
        nb_cells=NB_CELLS,
        rac="~/OUTPUT_MOLONARI1D/generated_data",  # printing directory by default,
        verbose=False,
    ):
        self._dir_print = create_dir(
            rac, verbose=verbose
        )  # once validated verbose=False

        self._classType = ClassType.COLUMN

        # ! Pour l'instant on suppose que les temps matchent
        self._times = [t for t, _ in dH_measures]
        # récupère la liste des charges de la riviière (au cours du temps)
        self._dH = np.array([d for _, (d, _) in dH_measures])
        # récupère la liste de température de la rivière (au cours du temps)
        self._T_riv = np.array([t for _, (_, t) in dH_measures])
        # récupère la liste de température de l'aquifère (au cours du temps)
        self._T_aq = np.array([t[-1] for _, t in T_measures])
        # récupère la liste de températures des capteurs (au cours du temps)
        self._T_measures = np.array([t[:-1] for _, t in T_measures])

        # décale d'un offset les positions des capteurs de température (aussi riviere)
        self._real_z = np.array([0] + depth_sensors) + offset
        # enlève l'offset sur la mesure de température rivière car cette mesure est prise dans le capteur pression
        self._real_z[0] -= offset

        self.depth_sensors = depth_sensors
        self.offset = offset

        self.zbed = river_bed  # profondeur du lit de la rivière

        self.all_layers = []  # liste des couches de la colonne
        if all_layers:  # layers can be given here or later with set_layers (B-20)
            self.set_layers(all_layers)

        self._z_solve = (
            None  # le tableau contenant la profondeur du milieu des cellules
        )
        self._id_sensors = None
        # le tableau contenant les températures à tout temps et à toute profondeur (lignes : températures) (colonnes : temps)
        self._temperatures = None

        # le tableau contenant les charges à tout temps et à toute profondeur (lignes : charges) (colonnes : temps)
        self._H_res = None
        # le tableau contenant le débit spécifique à tout temps et à toute profondeur (lignes : débit) (colonnes : temps)
        self._flows = None
        # le tableau contenant le flux advectif latéral (pseudo2D) à tout temps et à toute profondeur (lignes : flux) (colonnes : temps)
        self._lateral_advec_heat_flux = None

        self.tests([t for t, _ in T_measures])  # teste que les conditions nécessaires à l'analyse sont remplies
        # températures connues au premier temps; un capteur sans mesure (NaN) est laissé de côté
        T_first = np.array([self._T_riv[0], *self._T_measures[0], self._T_aq[0]])
        measured = ~np.isnan(T_first)
        self.lagr = Lagrange(
            self._real_z[measured], T_first[measured]
        )  # crée le polynome interpolateur de lagrange faisant coincider les températures connues à la profondeur réelle
        self.linear = interp1d(self._real_z[measured], T_first[measured])
        # crée la fonction affine par morceaux faisant coincider les températures connues à la profondeur réelle
        self.inter_mode = inter_mode
        self.eps = eps
        if verbose:
            print("Column created with success")
            print(f"Number of time steps: {self.get_timelength()}")
            print(f"Time step in days: {self.get_dt_in_days()}")
            print(f"T_riv: {self._T_riv}")
            print(f"T_aq: {self._T_aq}")
            print(f"dH : {self._dH}")
            print(f"list of dates   : {self._times}")
        self.initialization(nb_cells)

    def set_layers(
        self, layer: Union[Layer, List[Layer]]
    ):  # instancie une couche ou une liste de couches à la colonne, en s'assurant de bien ordonner les couches
        self.all_layers = []  # On vide la liste des couches avant d'en ajouter de nouvelles
        if isinstance(layer, Layer):
            self.all_layers.append(layer)
        elif isinstance(layer, list) and all(isinstance(l, Layer) for l in layer):
            self.all_layers.extend(layer)
        else:
            raise ValueError(
                "You must provide a Layer object or a list of Layer objects."
            )

        self.all_layers = sorted(
            self.all_layers, key=lambda x: x.zLow
        )  # we're sorting the layerlist by increasing zLow

    def _check_layers_geometry(self):
        # the layers must fill the column from the river bed down to its bottom, the deepest sensor (B-19)
        if not self.all_layers:
            raise ValueError("No layer in the column: call set_layers before computing.")
        bottom = self._real_z[-1]
        for layer in self.all_layers[:-1]:
            if not self._real_z[0] < layer.zLow < bottom - EPSILON:
                raise ValueError(
                    f"The boundary below layer '{layer.name}' (zLow = {layer.zLow} m) is outside the column, "
                    f"whose bottom (deepest sensor, offset included) is at {bottom} m."
                )
        last = self.all_layers[-1]
        if last.zLow < bottom - EPSILON:
            warnings.warn(
                f"The last layer '{last.name}' ends at zLow = {last.zLow} m, above the column bottom ({bottom} m): "
                f"it is extended down to the bottom.",
                UserWarning,
            )

    def initialization(self, nb_cells):
        self._nb_cells = nb_cells

    def tests(self, T_dates):
        """Checks the measurements before any computation. The dates of dH_measures and T_measures must be the
        same and strictly increasing, and the boundary conditions (head differential, river and aquifer
        temperatures) are needed at every time. Missing temperatures at the sensors (NaN) are left out of the
        initial profile, the energy of the MCMC and the RMSE, with a warning."""
        if len(T_dates) != len(self._times):
            raise ValueError(
                f"dH_measures and T_measures must have the same number of times, got {len(self._times)} and "
                f"{len(T_dates)}."
            )
        for j in range(1, len(self._times)):
            if not self._times[j] > self._times[j - 1]:
                raise ValueError(
                    f"The dates of the measurements must be strictly increasing: date {j} ({self._times[j]}) is "
                    f"not after date {j - 1} ({self._times[j - 1]})."
                )
        for j, (date_dH, date_T) in enumerate(zip(self._times, T_dates)):
            if date_dH != date_T:
                raise ValueError(
                    f"dH_measures and T_measures must have the same dates; they differ at index {j}: {date_dH} "
                    f"and {date_T}."
                )
        if self._T_measures.shape[1] + 1 != len(self.depth_sensors):
            raise ValueError(
                f"Each value of T_measures must have one temperature per depth of depth_sensors, the last one for "
                f"the aquifer: {len(self.depth_sensors)} depths, but {self._T_measures.shape[1] + 1} temperatures."
            )
        for values, name in (
            (self._dH, "head differential (dH_measures)"),
            (self._T_riv, "river temperature (dH_measures)"),
            (self._T_aq, "aquifer temperature (last value of T_measures)"),
        ):
            if np.isnan(values).any():
                raise ValueError(
                    f"Missing values (NaN) in the {name}: the boundary conditions are needed at every time."
                )
        missing = np.isnan(self._T_measures)
        if missing.any():
            empty = [f"{z} m" for z, none in zip(self.depth_sensors, missing.all(axis=0)) if none]
            warnings.warn(
                f"{missing.sum()} of {missing.size} temperature measurements at the sensors are missing (NaN); "
                f"they are left out of the initial profile, the energy of the MCMC and the RMSE."
                + (f" Sensors without any measurement: {', '.join(empty)}." if empty else ""),
                UserWarning,
            )

    @classmethod
    def from_dict(cls, col_dict, verbose=False):
        """
        Class method to create an instance of Column from a dictionnary.
        """
        return cls(**col_dict, verbose=verbose)

    ##########################################################################################"
    #
    #   Time management of the simulation
    #
    # #######################################################################################"

    def get_timelength(self):
        return len(self._times)

    def get_dt(self):
        # mean time step = mean of the actual intervals between successive measurements
        steps = np.diff(self._times)
        dt = np.mean([step.total_seconds() for step in steps])
        return dt

    def get_dt_in_days(self):
        dt = self.get_dt()
        return dt / NSECINDAY

    def create_time_in_day(self):
        nd = self.get_dt_in_days()
        return np.array([i * nd for i in range(self.get_timelength())])

    ##########################################################################################"
    #
    #   Computation function
    #
    # #######################################################################################"

    @checker
    def compute_solve_transi(self, verbose=True):
        self._check_layers_geometry()
        layersList = self.all_layers
        nb_cells = self._nb_cells

        if len(layersList) == 1:
            layer = layersList[
                0
            ]  # Cas homogène, formalisé en une seule couche faisant la taille de la colonne
            dz = self._real_z[-1] / nb_cells  # profondeur d'une cellule
            self._z_solve = dz / 2 + np.array(
                [k * dz for k in range(nb_cells)]
            )  # le tableau contenant la profondeur du milieu des cellules
            self._id_sensors = [
                np.argmin(np.abs(z - self._z_solve)) for z in self._real_z[1:-1]
            ]

            all_dt = np.array(
                [
                    (self._times[j + 1] - self._times[j]).total_seconds()
                    for j in range(len(self._times) - 1)
                ]
            )  # le tableau des pas de temps (dépend des données d'entrée)
            isdtconstant = np.all(all_dt == all_dt[0])

            H_init = self._dH[0] - self._dH[0] * self._z_solve / self._real_z[-1]
            # fixe toutes les charges de l'aquifère à 0 (à tout temps)
            H_aq = np.zeros(len(self._times))

            H_riv = self._dH  # self.dH contient déjà les charges de la rivière à tout temps, stocke juste dans une variable locale

            # crée les températures initiales (t=0) sur toutes les profondeurs (milieu des cellules)
            if self.inter_mode == "lagrange":
                T_init = np.array([self.lagr(z) for z in self._z_solve])
            elif self.inter_mode == "linear":
                T_init = self.linear(self._z_solve)

            T_riv = self._T_riv
            T_aq = self._T_aq

            IntrinK, n, lambda_s, rhos_cs, q_s = layer.get_physical_params()
            if verbose:
                print(
                    "--- Compute Solve Transi ---",
                    f"One layer : IntrinK = {IntrinK}, n = {n}, lambda_s = {lambda_s}, rhos_cs = {rhos_cs}, q_s = {q_s}",
                    sep="\n",
                )

            heigth = abs(self._real_z[-1] - self._real_z[0])
            Ss = n / heigth  # l'emmagasinement spécifique = porosité sur la hauteur

            ## pour le cas uni-couche, on le simule dans H_stratified avec deux couches de mêmes paramètres
            array_IntrinK = np.array([IntrinK, IntrinK])
            array_K = calc_K(array_IntrinK)
            array_Ss = np.array([Ss, Ss])
            list_zLow = np.array([0.2])
            inter_cara = np.array([[nb_cells // 2, 0]])
            z_solve = self._z_solve.copy()
            IntrinK_list, n_list, lambda_s_list, rhos_cs_list, q_s_list = (
                getListParameters(layersList, self._z_solve)
            )
            Ss_list = n_list / heigth

            ##

            H_strat = H_stratified(
                Ss_list,
                IntrinK_list,
                n_list,
                lambda_s_list,
                rhos_cs_list,
                all_dt,
                q_s_list,
                dz,
                H_init,
                H_riv,
                H_aq,
                T_init,
                array_K,
                array_Ss,
                list_zLow,
                z_solve,
                inter_cara,
                isdtconstant,
                alpha=ALPHA,
            )
            H_res = H_strat.compute_H_stratified()

            nablaH = H_strat.nablaH()

            T_strat = T_stratified(
                nablaH,
                Ss_list,
                IntrinK_list,
                n_list,
                lambda_s_list,
                rhos_cs_list,
                all_dt,
                q_s_list,
                dz,
                H_init,
                H_riv,
                H_aq,
                T_init,
                T_riv,
                T_aq,
                alpha=ALPHA,
                N_update_Mu=N_UPDATE_MU,
            )
            T_res = T_strat.compute_T_stratified()

            # on récupère le flux de chaleur latéral (W.m-3) et on convertit en (W.m-2) en multipliant par dz
            if hasattr(T_strat, "source_heat_flux"):
                self._lateral_advec_heat_flux = T_strat.source_heat_flux * dz

            # calcule toutes les températures à tout temps et à toute profondeur

            self._temperatures = T_res
            self._H_res = H_res  # stocke les résultats

            K = calc_K(
                IntrinK
            )  # NF it will need to be changed with the dependency to temperature
            if verbose:
                print(
                    f"Solving the flow with intrinsec permeability {IntrinK}, and permeability {K}"
                )
            self._flows = -K * nablaH  # calcul du débit spécifique

            if verbose:
                print("Done.")

        else:  # case multiple layers
            dz = self._real_z[-1] / nb_cells  # profondeur d'une cellule
            self._z_solve = dz / 2 + np.array([k * dz for k in range(nb_cells)])

            self._id_sensors = [
                np.argmin(np.abs(z - self._z_solve)) for z in self._real_z[1:-1]
            ]

            all_dt = np.array(
                [
                    (self._times[j + 1] - self._times[j]).total_seconds()
                    for j in range(len(self._times) - 1)
                ]
            )  # le tableau des pas de temps (dépend des données d'entrée)
            isdtconstant = np.all(all_dt == all_dt[0])

            H_init = (
                self._dH[0] - self._dH[0] * self._z_solve / self._real_z[-1]
            )  # linear interpolation not usabe in multiple case
            # fixe toutes les charges de l'aquifère à 0 (à tout temps)
            H_aq = np.zeros(len(self._times))
            H_riv = self._dH  # self.dH contient déjà les charges de la rivière à tout temps, stocke juste dans une variable locale
            # crée les températures initiales (t=0) sur toutes les profondeurs (milieu des cellules)

            if self.inter_mode == "lagrange":
                T_init = np.array([self.lagr(z) for z in self._z_solve])
            elif self.inter_mode == "linear":
                T_init = self.linear(self._z_solve)

            T_riv = self._T_riv
            T_aq = self._T_aq

            IntrinK_list, n_list, lambda_s_list, rhos_cs_list, q_s_list = (
                getListParameters(layersList, self._z_solve)
            )

            array_IntrinK = np.array(
                [
                    float(layer.get_physical_params().IntrinK)
                    for layer in self.all_layers
                ]
            )
            array_K = calc_K(array_IntrinK)
            heigth = abs(self._real_z[-1] - self._real_z[0])
            array_Ss = (
                np.array([float(x.get_physical_params().n) for x in layersList])
                / heigth
            )
            # thickness of each layer inside the column; the last layer ends at the column bottom (B-19)
            z_bounds = [layer.zLow for layer in layersList[:-1]] + [self._real_z[-1]]
            array_eps = np.diff(z_bounds, prepend=self._real_z[0])

            array_Hinter = np.zeros(
                len(layersList) + 1
            )  # charge hydraulique de chaque interface
            array_Hinter[0] = self._dH[0]
            array_Hinter[-1] = 0.0
            N = len(array_Hinter) - 1
            # calculate Hinter
            H_gauche = np.zeros((N - 1, N - 1))
            H_droite = np.zeros((N - 1, N - 1))
            scalar_gauche = np.zeros(N - 1)
            scalar_droite = np.zeros(N - 1)
            scalar_gauche[0] = array_K[0] * array_Hinter[0] / array_eps[0]
            scalar_droite[-1] = -array_K[-1] * array_Hinter[-1] / array_eps[-1]
            H_gauche[0, 0] = -array_K[0] / array_eps[0]
            for diag in range(1, N - 1):
                H_gauche[diag, diag - 1] = array_K[diag] / array_eps[diag]
                H_gauche[diag, diag] = -array_K[diag] / array_eps[diag]

            H_droite[N - 2, N - 2] = array_K[-1] / array_eps[-1]
            for diag in range(0, N - 2):
                H_droite[diag, diag + 1] = -array_K[diag + 1] / array_eps[diag + 1]
                H_droite[diag, diag] = array_K[diag + 1] / array_eps[diag + 1]

            Matrix_b = scalar_gauche - scalar_droite
            Matrix_A = H_droite - H_gauche
            H_sol = np.linalg.solve(Matrix_A, Matrix_b)
            for idx in range(len(H_sol)):
                array_Hinter[idx + 1] = H_sol[idx]
            #
            # list_array_L: couper le profondeur selon l'épaisseur de chaque couche
            list_array_L = []
            cnt = 0
            for idx in range(len(layersList) - 1):
                cnt_start = cnt
                while cnt < len(self._z_solve):
                    if (
                        self._z_solve[cnt] <= layersList[idx].zLow
                        and self._z_solve[cnt + 1] > layersList[idx].zLow
                    ):
                        list_array_L.append(self._z_solve[cnt_start : cnt + 1])
                        cnt += 1
                        break
                    else:
                        cnt += 1
            list_array_L.append(self._z_solve[cnt:])

            #
            # calculer H de chaque couche

            list_array_H = []
            for idx in range(len(list_array_L)):
                if idx > 0:
                    list_array_H.append(
                        array_Hinter[idx]
                        - (array_Hinter[idx] - array_Hinter[idx + 1])
                        / array_eps[idx]
                        * (list_array_L[idx] - layersList[idx - 1].zLow)
                    )
                else:
                    list_array_H.append(
                        array_Hinter[idx]
                        - (array_Hinter[idx] - array_Hinter[idx + 1])
                        / array_eps[idx]
                        * (list_array_L[idx] - 0)
                    )
    
            H_init = list_array_H[0]
            for idx in range(1, len(list_array_H)):
                H_init = np.concatenate((H_init, list_array_H[idx]))

            Ss_list = (
                n_list / heigth
            )  # l'emmagasinement spécifique = porosité sur la hauteur

            if verbose:
                print("--- Compute Solve Transi ---")
                for layer in layersList:
                    print(layer)
                print("Hinter", array_Hinter)

            if verbose:
                print("conditions aux limites")
                print("H_riv", H_riv)
                print("H_aq", H_aq)

            list_zLow = []
            for layer in layersList:
                list_zLow.append(layer.zLow)
            list_zLow.pop()
            list_zLow = np.array(list_zLow)
            z_solve = self._z_solve.copy()

            ## Classification according to Klist on symmetry of interfaces
            inter_cara = np.zeros((len(list_zLow), 2))
            for zlow_idx in range(len(list_zLow)):
                for z_idx in range(len(z_solve) - 1):
                    if (
                        z_solve[z_idx] <= list_zLow[zlow_idx]
                        and z_solve[z_idx + 1] > list_zLow[zlow_idx]
                    ):
                        if verbose:
                            print(
                                "échantillons du profondeur: ... ",
                                z_solve[z_idx],
                                z_solve[z_idx + 1],
                                " ...",
                            )
                            print("le profondeur d'interface: ", list_zLow[zlow_idx])
                        if abs(z_solve[z_idx] - list_zLow[zlow_idx]) < EPSILON:
                            inter_cara[zlow_idx, 0] = z_idx
                            if verbose:
                                print("type cara symetric")
                        elif abs(z_solve[z_idx + 1] - list_zLow[zlow_idx]) < EPSILON:
                            inter_cara[zlow_idx, 0] = z_idx + 1
                            if verbose:
                                print("type cara symetric")
                        else:
                            inter_cara[zlow_idx, 0] = z_idx
                            inter_cara[zlow_idx, 1] = z_idx + 1
                            if verbose:
                                print("type cara asymetric")
            ## end
            # version zhan
            a = 1  # à adapter
            H_strat = H_stratified(
                Ss_list,
                IntrinK_list,
                n_list,
                lambda_s_list,
                rhos_cs_list,
                all_dt,
                q_s_list,
                dz,
                H_init,
                H_riv,
                H_aq,
                T_init,
                array_K,
                array_Ss,
                list_zLow,
                z_solve,
                inter_cara,
                isdtconstant,
                alpha=ALPHA,
            )
            H_res = H_strat.compute_H_stratified()

            self._H_res = H_res  # stocke les résultats

            # ###
            # H_res = HTK_stratified(a,Ss_list,IntrinK_list,n_list,lambda_s_list,rhos_cs_list,all_dt,dz,H_init,H_riv,H_aq,T_init,T_riv,T_aq,array_K,array_Ss,list_zLow,z_solve,inter_cara,isdtconstant,q_s_list,alpha=ALPHA,N_update_Mu=N_UPDATE_MU,).compute_HTK_stratified()
            # ###

            # création d'un tableau du gradient de la charge selon la profondeur, calculé à tout temps
            K_list = calc_K(IntrinK_list)

            nablaH = H_strat.nablaH()

            flows = np.zeros((nb_cells, len(self._times)), np.float64)

            for i in range(nb_cells):
                flows[i, :] = -K_list[i] * nablaH[i, :]

            for elem_idx in range(len(inter_cara)):
                if inter_cara[elem_idx][1] == 0:
                    if (
                        K_list[int(inter_cara[elem_idx][0])]
                        == K_list[int(inter_cara[elem_idx][0]) + 1]
                    ):
                        nablaH[int(inter_cara[elem_idx][0]), :] = nablaH[
                            int(inter_cara[elem_idx][0]) + 1, :
                        ]  ## +1
                    if (
                        K_list[int(inter_cara[elem_idx][0])]
                        == K_list[int(inter_cara[elem_idx][0]) - 1]
                    ):
                        nablaH[int(inter_cara[elem_idx][0]), :] = nablaH[
                            int(inter_cara[elem_idx][0]) - 1, :
                        ]  ## +1

                    flows[int(inter_cara[elem_idx][0]), :] = (
                        -K_list[int(inter_cara[elem_idx][0])]
                        * nablaH[int(inter_cara[elem_idx][0]), :]
                    )
                else:
                    nablaH[int(inter_cara[elem_idx][0]), :] = (
                        H_res[int(inter_cara[elem_idx][0]) + 1, :]
                        - H_res[int(inter_cara[elem_idx][0]), :]
                    ) / dz
                    nablaH[int(inter_cara[elem_idx][1]), :] = (
                        H_res[int(inter_cara[elem_idx][1]), :]
                        - H_res[int(inter_cara[elem_idx][1]) - 1, :]
                    ) / dz
                    x = (
                        list_zLow[elem_idx] - z_solve[int(inter_cara[elem_idx][0])]
                    ) / (
                        z_solve[int(inter_cara[elem_idx][1])]
                        - z_solve[int(inter_cara[elem_idx][0])]
                    )
                    K_list[int(inter_cara[elem_idx][0])] = 1 / (
                        x / K_list[int(inter_cara[elem_idx][0])]
                        + (1 - x) / K_list[int(inter_cara[elem_idx][0]) + 1]
                    )
                    flows[int(inter_cara[elem_idx][0]), :] = (
                        -K_list[int(inter_cara[elem_idx][0])]
                        * nablaH[int(inter_cara[elem_idx][0]), :]
                    )
                    flows[int(inter_cara[elem_idx][1]), :] = (
                        -K_list[int(inter_cara[elem_idx][0])]
                        * nablaH[int(inter_cara[elem_idx][1]), :]
                    )

            ## zhan Nov8
            T_strat = T_stratified(
                nablaH,
                Ss_list,
                IntrinK_list,
                n_list,
                lambda_s_list,
                rhos_cs_list,
                all_dt,
                q_s_list,
                dz,
                H_init,
                H_riv,
                H_aq,
                T_init,
                T_riv,
                T_aq,
                alpha=ALPHA,
                N_update_Mu=N_UPDATE_MU,
            )
            T_res = T_strat.compute_T_stratified()
            self._temperatures = T_res

            self._flows = flows  # calcul du débit spécifique
            if verbose:
                print("Done.")

            if np.isnan(self._temperatures).any() or np.isnan(self._flows).any():
                print(
                    f"Issue for the following parameters : {self.get_list_current_params()}"
                )
                print(
                    f"Issue for the follwing number of layers : {len(self.all_layers)}"
                )
                raise ValueError("NaN values in compute_solve_transi")

    @compute_solve_transi.needed
    def get_id_sensors(self):
        """
        Returns
        -------
        self._id_sensors : int list
            list of the 3 indices of the cells where the non boundary sensors are.
        """
        return self._id_sensors

    @compute_solve_transi.needed
    def get_RMSE(self):
        """
        Returns
        -------
        res : float array
            array with 4 elements which contains the RMSE for the non boundary sensors, and the total RMSE
        """

        # Array of RMSE for each sensor (except boundary conditions : river and aquifer), over its measured times
        list_RMSE = np.array(
            [
                rmse_measured(self.get_temperatures_solve()[id, :], temperatures_obs)
                for id, temperatures_obs in zip(
                    self.get_id_sensors(), self._T_measures.T
                )
            ]
        )

        # Total RMSE, over the sensors with measurements
        total_RMSE = np.sqrt(np.mean(list_RMSE[~np.isnan(list_RMSE)] ** 2))

        return np.append(list_RMSE, total_RMSE)

    @compute_solve_transi.needed
    def print_RMSE_at_sensor(self):
        rmse = self.get_RMSE()
        for i in range(len(rmse) - 1):
            print(f"RMSE at sensor {i} : {rmse[i]}")
        print(f"Total RMSE : {rmse[-1]}")

    # erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    @compute_solve_transi.needed
    def get_depths_solve(self):
        """
        Returns
        -------
        self._z_solve : float array
            array of the depths of the middle of each cell.
        """
        return self._z_solve

    depths_solve = property(get_depths_solve)
    # récupération de l'attribut _z_solve

    def get_times_solve(self):
        """
        Returns
        -------
        self._times : datetime list
            list of the times at which the temperatures are computed.
        """
        return self._times

    times_solve = property(get_times_solve)
    # récupération de l'attribut _times

    # erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    @compute_solve_transi.needed
    def get_temperatures_solve(self, z=None):
        if z is None:
            return self._temperatures
        z_ind = np.argmin(np.abs(self.depths_solve - z))
        return self._temperatures[z_ind, :]

    temperatures_solve = property(get_temperatures_solve)
    # récupération des températures au cours du temps à toutes les profondeurs (par défaut) ou bien à une profondeur donnée

    # erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    # on renomme le getter pour distinguer l'advectif vertical et latéral
    @compute_solve_transi.needed
    def get_vertical_advec_flows_solve(self):
        return RHO_W * C_W * self._flows * (self.temperatures_solve - ZERO_CELSIUS)

    vertical_advec_flows_solve = property(get_vertical_advec_flows_solve)
    # récupération des flux advectifs = masse volumnique*capacité calorifique*débit spécifique*température

    # nouveau getter
    def get_lateral_advec_heat_flux_solve(self, z=None):
        """
        Retourne le flux de chaleur surfacique (W/m2) généré par le terme source latéral.
        """
        if z is None:
            return self._lateral_advec_heat_flux

        z_ind = np.argmin(np.abs(self.depths_solve - z))
        return self._lateral_advec_heat_flux[z_ind, :]

    lateral_advec_heat_flux_solve = property(get_lateral_advec_heat_flux_solve)

    # erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    @compute_solve_transi.needed
    def get_conduc_flows_solve(self):
        dz = self._z_solve[1] - self._z_solve[0]  # pas en profondeur
        nb_cells = len(self._z_solve)

        _, n_list, lambda_s_list, _, _ = getListParameters(self.all_layers, self._z_solve)

        lambda_m_list = (
            n_list * (LAMBDA_W) ** 0.5 + (1.0 - n_list) * (lambda_s_list) ** 0.5
        ) ** 2  # conductivité thermique du milieu poreux équivalent

        # création du gradient de température
        nablaT = np.zeros((nb_cells, len(self._times)), np.float64)

        for i in range(1, nb_cells - 1):
            nablaT[i, :] = (
                self._temperatures[i + 1, :] - self._temperatures[i - 1, :]
            ) / (2 * dz)
        nablaT[0, :] = nablaT[1, :]
        nablaT[nb_cells - 1, :] = nablaT[nb_cells - 2, :]

        conduc_flows = np.zeros((nb_cells, len(self._times)), np.float64)

        for i in range(nb_cells):
            conduc_flows[i, :] = lambda_m_list[i] * nablaT[i, :]

        return conduc_flows

    conduc_flows_solve = property(get_conduc_flows_solve)
    # récupération des flux conductifs = conductivité*gradient(T)

    # erreur si pas déjà éxécuté compute_solve_transi, sinon l'attribut pas encore affecté à une valeur
    @compute_solve_transi.needed
    def get_flows_solve(self, z=None):
        if z is None:
            return self._flows  # par défaut, retourne le tableau des débits spécifiques
        z_ind = np.argmin(np.abs(self.depths_solve - z))
        # sinon ne les retourne que pour la profondeur choisie
        return self._flows[z_ind, :]

    flows_solve = property(get_flows_solve)
    # récupération des débits spécifiques au cours du temps à toutes les profondeurs (par défaut) ou bien à une profondeur donnée

    @compute_solve_transi.needed
    def get_temperature_at_sensors(
        self, zero=0, verbose=False
    ):  # possibility to translate temperature with zero=ZERO_CELSIUS. as is temperature in Kelvin
        ids = self.get_id_sensors()
        if verbose:
            print(f"{len(ids)} Sensors\n")
        temperatures = np.zeros(
            (len(ids) + 2, self.get_timelength())
        )  # adding the boundary conditions
        temperatures[0] = self._T_riv - zero
        temperatures[len(ids) + 1] = self._T_aq - zero
        for id in range(len(ids)):
            temperatures[id + 1] = self.get_temperatures_solve()[ids[id]] - zero
        return temperatures

    @compute_solve_transi.needed
    def plot_it_Zt(
        self,
        plotIt,
        title="TBD",
        cbarUnits="TBD",
        distBot=1.0,
        distLeft=0.5,
        fontsize=15,
    ):
        zoomSize = 2
        titleSize = fontsize + zoomSize
        temps_en_jours = self.create_time_in_day()
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="w")

        """Plots des profils de température"""

        im = ax.imshow(
            plotIt,
            aspect="auto",
            extent=[
                temps_en_jours[0],
                temps_en_jours[-1],
                self.depths_solve[-1],
                self.depths_solve[0],
            ],
            cmap="Spectral_r",
        )

        ax.set_xlabel("time in days", fontsize=fontsize)
        ax.set_ylabel("depth in m", fontsize=fontsize)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position("bottom")

        cbar = plt.colorbar(im, ax=ax, shrink=1, location="right")

        cbarSize = ax.xaxis.get_ticklabels()[
            0
        ].get_fontsize()  # Récupère la taille de police des ticks de l'axe
        cbar.ax.tick_params(
            labelsize=cbarSize
        )  # Applique la même taille de police aux ticks de la colorbar
        cbar.ax.xaxis.set_label_position("top")  # Positionne l'étiquette en haut
        cbar.ax.xaxis.label.set_rotation(
            0
        )  # Garder l'étiquette horizontale (rotation 0°)
        # Ajouter le titre de la colorbar manuellement, en utilisant text()
        cbar.ax.text(
            distLeft,
            distBot,
            cbarUnits,
            ha="center",
            va="bottom",
            transform=cbar.ax.transAxes,
            fontsize=fontsize,
        )

        ax.set_title(title, fontsize=titleSize)
        plt.show()

    @compute_solve_transi.needed
    def plot_temperature_at_sensors(
        self, title="Temperatures", verbose=False, fontsize=15
    ):
        zoomSize = 2
        titleSize = fontsize + zoomSize
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="w")

        temps_en_jours = self.create_time_in_day()
        ids = self.get_id_sensors()
        temperatures = self.get_temperature_at_sensors(verbose=verbose)

        for i in range(len(ids) + 2):
            if verbose:
                print(f"Plotting Sensor {i}\n")
            if i == 0:
                label = "T_riv"
            elif i == len(ids) + 1:
                label = "T_aq"
            else:
                label = f"T{i}"

            ax.plot(temps_en_jours, temperatures[i], label=label)

        ax.set_xlabel("time in days", fontsize=fontsize)
        ax.set_ylabel("temperature", fontsize=fontsize)
        ax.set_title(title, fontsize=titleSize)
        # Add legend on the right side with the same font size as axis tick labels
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize - zoomSize
        )

        plt.show()

    def set_zeroT(self, tunits="K"):
        if tunits == "K":
            zeroT = 0
        else:
            zeroT = ZERO_CELSIUS
        return zeroT

    @compute_solve_transi.needed
    def plot_compare_temperatures_sensors(self, tunits="K", fontsize=15):
        zeroT = self.set_zeroT(tunits)
        temps_en_jours = self.create_time_in_day()
        ids = self.get_id_sensors()
        temperatures = self.get_temperature_at_sensors() - zeroT
        nsensors = len(ids)
        fig, axes = plt.subplots(1, nsensors, figsize=(20, 5), sharey=True)
        axes[0].set_ylabel(f"Temperature in {tunits}")

        for i in range(nsensors):
            axes[i].set_xlabel("time in days", fontsize=fontsize)
            axes[i].plot(
                temps_en_jours,
                self._T_measures[:, i] - zeroT,
                label="Measurement",
            )
            axes[i].plot(temps_en_jours, temperatures[i + 1], label="Simulated")
            axes[i].legend()
            axes[i].set_title(f"Sensor {i + 1}")

        plt.subplots_adjust(wspace=0.05)

    @compute_solve_transi.needed
    def plot_temperatures_umbrella(self, dplot=1, K_offset=0, fontsize=15):
        K_offset = ZERO_CELSIUS
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="w")
        nt = len(self._times)

        # Create a custom colormap from light orange to dark brown
        colors = [(1, 0.8, 0.6), (0.4, 0.2, 0)]  # Light orange to dark brown
        n_bins = nt // dplot  # Discretize the colormap
        cmap_name = "orange_brown"
        colormap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)
        # Generate colors from the custom colormap
        colors = [colormap(i / (nt // dplot)) for i in range(nt // dplot)]

        # Define line styles
        line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 10))]

        for idx, i in enumerate(range(0, nt, dplot)):
            color = colors[idx % len(colors)]
            linestyle = line_styles[idx % len(line_styles)]
            ax.plot(
                self._temperatures[:nt, i] - K_offset,
                -self._z_solve,
                label=f"T@{i * self.get_dt_in_days():.3f}d",
                color=color,
                linestyle=linestyle,
            )

        # Add legend with specified font size
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=fontsize)
        # set labels and title
        ax.set_ylabel("Depth (m)", fontsize=fontsize)
        ax.set_xlabel("T (°C)", fontsize=fontsize)
        ax.grid()
        ax.set_title("Temperature profiles over time", fontsize=fontsize, pad=20)
        # Display the plot
        plt.tight_layout()
        plt.show()

    @compute_solve_transi.needed
    def plot_CALC_results(self, fontsize=15):
        print(
            f"Plotting Température in column. time series have nrecords =  {len(self._times)}"
        )
        nt = len(self._times)
        K_offset = ZERO_CELSIUS
        n_sens = len(self.depth_sensors) - 1

        """Changement de l'échelle de l'axe x"""

        def min2jour(x):
            return x * self.get_dt_in_days()

        def jour2min(x):
            return x / self.get_dt_in_days()

        """Plots des profils de température"""

        # Créer une grille 3x3 et ajuster la taille
        fig, ax = plt.subplots(3, 3, sharey=False, figsize=(22, 21))
        plt.subplots_adjust(wspace=0.3, hspace=0.4)
        fig.suptitle("Résultats calcul : simulateur de données", fontsize=fontsize + 6)

        # LIGNE 1, PLOT 1 : Températures mesurées
        ax[0, 0].plot(self._T_riv[:nt][:nt] - K_offset, label="Triv")
        for i in range(n_sens):
            ax[0, 0].plot(
                self._T_measures[:nt, i] - K_offset, label="T{}".format(i + 1)
            )
        ax[0, 0].plot(self._T_aq[:nt] - K_offset, label="Taq")
        ax[0, 0].legend(fontsize=fontsize)
        ax[0, 0].grid()
        ax[0, 0].xaxis.tick_top()
        ax[0, 0].set_xlabel("n*dt (15min)", fontsize=fontsize)
        ax[0, 0].xaxis.set_label_position("top")
        ax[0, 0].set_ylabel("T (°C)", fontsize=fontsize)
        ax[0, 0].secax = ax[0, 0].secondary_xaxis(
            "bottom", functions=(min2jour, jour2min)
        )
        ax[0, 0].secax.set_xlabel("t (jour)", fontsize=fontsize)
        ax[0, 0].set_title("Température mesurées", fontsize=fontsize, pad=20)

        # LIGNE 1, PLOT 2 : Evolution du profil de température
        for i in range(nt):
            ax[0, 1].plot(self._temperatures[:nt, i] - K_offset, -self._z_solve)
        ax[0, 1].set_ylabel("Depth (m)", fontsize=fontsize)
        ax[0, 1].set_xlabel("T (°C)", fontsize=fontsize)
        ax[0, 1].grid()
        ax[0, 1].set_title(
            "Evolution du profil de température", fontsize=fontsize, pad=20
        )

        #  Masquer l'axe inutilisé de la première ligne
        ax[0, 2].axis("off")

        # LIGNE 2, PLOT 1 : Frise température
        im0 = ax[1, 0].imshow(
            self._temperatures[:, :nt] - K_offset, aspect="auto", cmap="Spectral_r"
        )
        ax[1, 0].set_xlabel("t (15min)", fontsize=fontsize)
        ax[1, 0].set_ylabel("z (m)", fontsize=fontsize)
        ax[1, 0].xaxis.tick_top()
        ax[1, 0].xaxis.set_label_position("top")
        ax[1, 0].secax = ax[1, 0].secondary_xaxis(
            "bottom", functions=(min2jour, jour2min)
        )
        ax[1, 0].secax.set_xlabel("t (jour)", fontsize=fontsize)
        cbar0 = fig.colorbar(im0, ax=ax[1, 0], shrink=1, location="right")
        cbar0.set_label("Température (°C)", fontsize=fontsize)
        ax[1, 0].set_title("Frise température MD", fontsize=fontsize, pad=20)

        # LIGNE 2, PLOT 2 : Frise Flux d'eau
        im3 = ax[1, 1].imshow(
            self.get_flows_solve()[:, :nt], aspect="auto", cmap="Spectral_r"
        )
        ax[1, 1].set_xlabel("t (15min)", fontsize=fontsize)
        ax[1, 1].set_ylabel("z (m)", fontsize=fontsize)
        ax[1, 1].xaxis.tick_top()
        ax[1, 1].xaxis.set_label_position("top")
        ax[1, 1].secax = ax[1, 1].secondary_xaxis(
            "bottom", functions=(min2jour, jour2min)
        )
        ax[1, 1].secax.set_xlabel("t (jour)", fontsize=fontsize)
        cbar3 = fig.colorbar(im3, ax=ax[1, 1], shrink=1, location="right")
        cbar3.set_label("Water flow (m/s)", fontsize=fontsize)
        ax[1, 1].set_title("Frise Flux d'eau MD", fontsize=fontsize, pad=20)

        # Masquer l'axe inutilisé de la deuxième ligne
        ax[1, 2].axis("off")

        # LIGNE 3, PLOT 1 : Frise Flux conductif
        im1 = ax[2, 0].imshow(
            self.get_conduc_flows_solve()[:, :nt], aspect="auto", cmap="Spectral_r"
        )
        ax[2, 0].set_xlabel("t (15min)", fontsize=fontsize)
        ax[2, 0].set_ylabel("z (m)", fontsize=fontsize)
        ax[2, 0].xaxis.tick_top()
        ax[2, 0].xaxis.set_label_position("top")
        ax[2, 0].secax = ax[2, 0].secondary_xaxis(
            "bottom", functions=(min2jour, jour2min)
        )
        ax[2, 0].secax.set_xlabel("t (jour)", fontsize=fontsize)
        cbar1 = fig.colorbar(im1, ax=ax[2, 0], shrink=1, location="right")
        cbar1.set_label("Flux conductif (W/m²)", fontsize=fontsize)
        ax[2, 0].set_title("Frise Flux conductif MD", fontsize=fontsize, pad=20)

        # LIGNE 3, PLOT 2 : Frise Flux advectif vertical
        im2 = ax[2, 1].imshow(
            self.get_vertical_advec_flows_solve()[:, :nt],
            aspect="auto",
            cmap="Spectral_r",
        )
        ax[2, 1].set_xlabel("t (15min)", fontsize=fontsize)
        ax[2, 1].set_ylabel("z (m)", fontsize=fontsize)
        ax[2, 1].xaxis.tick_top()
        ax[2, 1].xaxis.set_label_position("top")
        ax[2, 1].secax = ax[2, 1].secondary_xaxis(
            "bottom", functions=(min2jour, jour2min)
        )
        ax[2, 1].secax.set_xlabel("t (jour)", fontsize=fontsize)
        cbar2 = fig.colorbar(im2, ax=ax[2, 1], shrink=1, location="right")
        cbar2.set_label("Vertical Advective Flux (W/m²)", fontsize=fontsize)
        ax[2, 1].set_title("Frise Flux advectif vertical MD", fontsize=fontsize, pad=20)

        # LIGNE 3, PLOT 3 : Frise Flux advectif latéral
        if self._lateral_advec_heat_flux is not None:
            im4 = ax[2, 2].imshow(
                self.get_lateral_advec_heat_flux_solve()[:, :nt],
                aspect="auto",
                cmap="Spectral_r",
            )
            ax[2, 2].set_xlabel("t (15min)", fontsize=fontsize)
            ax[2, 2].set_ylabel("z (m)", fontsize=fontsize)
            ax[2, 2].xaxis.tick_top()
            ax[2, 2].xaxis.set_label_position("top")
            ax[2, 2].secax = ax[2, 2].secondary_xaxis(
                "bottom", functions=(min2jour, jour2min)
            )
            ax[2, 2].secax.set_xlabel("t (jour)", fontsize=fontsize)
            cbar4 = fig.colorbar(im4, ax=ax[2, 2], shrink=1, location="right")
            cbar4.set_label("Lateral Advective Flux (W/m²)", fontsize=fontsize)
            ax[2, 2].set_title(
                "Frise Flux advectif latéral MD", fontsize=fontsize, pad=20
            )
        else:
            # S'il n'y a pas de flux latéral, on masque aussi ce dernier axe
            ax[2, 2].axis("off")

    @compute_solve_transi.needed
    def plot_all_results(self):
        self.print_RMSE_at_sensor()
        self.plot_compare_temperatures_sensors()
        self.plot_temperature_at_sensors()

        nt = len(self._temperatures[0, :])
        dplot = 15
        self.plot_temperatures_umbrella(round(nt / dplot))

        flows = self.get_flows_solve()
        unitLeg = "m/s"
        title = "Darcy velocity"
        self.plot_it_Zt(flows, title, unitLeg, 1.04, 2)

        ### 2.3 Other plots
        temperatures = self.get_temperatures_solve()
        unitLeg = "K"
        title = "Temperatures"
        self.plot_it_Zt(temperatures, title, unitLeg)

        flux_advectifs_verticaux = self.get_vertical_advec_flows_solve()
        unitLeg = "W/m2"
        title = "Vertical Advective heat flux"
        self.plot_it_Zt(flux_advectifs_verticaux, title, unitLeg, 1.04, 2)

        flux_conductifs = self.get_conduc_flows_solve()
        unitLeg = "W/m2"
        title = "Conductive heat flux"
        self.plot_it_Zt(flux_conductifs, title, unitLeg, 1.04, 2)

        if self._lateral_advec_heat_flux is not None:
            flux_advectif_lateral = self.get_lateral_advec_heat_flux_solve()
            unitLeg = "W/m2"
            title = "Lateral Advective Heat Flux"
            self.plot_it_Zt(flux_advectif_lateral, title, unitLeg, 1.04, 2)

        self.plot_CALC_results()

    def print_sensor_file(self, fp, senType, senName):
        print(f"Printing Sensor file {senType}")
        if senType == SensorType.pressure_sensors.value:
            fp.write(f"P_Sensor_Name,{senName}\n")
            fp.write(f"Datalogger,MOLONARI1D_by_arduino\n")
            fp.write(f"Calibration_Date,{ABSURD_DATE}\n")
            fp.write(f"Intercept,0.\n")
            fp.write(f"dU_dH,1\n")
            fp.write(f"dU_dT,0.0\n")
            fp.write(f"Sigma_Meas_P,0.01\n")
            fp.write(
                f"Thermometer name,{SENSOR_FILE_NAMES[SensorType.temperature_sensors]}\n"
            )
        elif senType == SensorType.shafts.value:
            fp.write(f"Shaft_Name,{senName}\n")
            fp.write(f"Datalogger,Hobo\n")
            fp.write(
                f"T_Sensor_Name,{SENSOR_FILE_NAMES[SensorType.temperature_sensors]}\n"
            )
            fp.write(f'Sensors_Depth,"{self.depth_sensors}"\n')

        else:
            fp.write(f"Cons_Name,Mines\n")
            fp.write(f"Cons_Ref,Corefge\n")
            fp.write(f"T_Sensor_Name,{senName}\n")
            fp.write(f"Sigma_Meas_T,0.03\n")

    def print_sampling_point_info(
        self,
        fp,
        spname="VirtualPoint",
        psenname="Pvirtual",
        shaftname="Svirtual",
    ):
        fp.write(f"Point_Name,{spname}\n")
        fp.write(f"P_Sensor_Name,{psenname}\n")
        fp.write(f"Shaft_Name,{shaftname}\n")
        fp.write(f"Implantation_Date,{self._times[0]}\n")
        fp.write(f"Meas_Date,{self._times[-1]}\n")
        fp.write(f"River_Bed,{self.zbed}\n")
        fp.write(f"Delta_h,{self.offset}\n")

    @compute_solve_transi.needed
    def print_in_file_processed_MOLONARI_dataset(
        self, zeroT=ZERO_CELSIUS, spname="VirtualPoint", lname="VirtualLabo"
    ):
        spDirName = f"{self._dir_print}/{spname}"
        pointDir = create_dir(spDirName)

        labDirName = f"{self._dir_print}/{lname}"
        labDir = create_dir(labDirName)

        ids = self.get_id_sensors()
        temperatures = self.get_temperature_at_sensors()

        fpressure = open_printable_file(
            rac=pointDir,
            dataType=DeviceType.PRESSURE,
            classType=self._classType,
            spname=spname,
        )
        fthermal = open_printable_file(
            rac=pointDir,
            dataType=DeviceType.TEMPERATURE,
            classType=self._classType,
            spname=spname,
        )

        for i in range(len(self._times)):
            # Format the datetime to the desired format
            formatted_time = self._times[i].strftime("%m/%d/%y %I:%M:%S %p")

            fpressure.write(
                f"{formatted_time},{self._dH[i]:.4f},{temperatures[0][i] - zeroT:.4f}\n"
            )
            # Initialize a list to store the temperature values
            temp_values = [f"{formatted_time}"]
            # Loop through each id in ids and append the corresponding temperature slice to the list
            for id in range(len(ids) + 1):
                temp_values.append(f"{temperatures[id + 1][i] - zeroT:.4f}")
                # Join the list elements into a single string separated by commas
                temp_string = ",".join(temp_values)
                # Write the formatted string to fthermal
            fthermal.write(
                f"{temp_string}\n"
            )  # Initialize a list to store the temperature values

        finfo = open_printable_file(rac=pointDir, fname=f"{spname}_info")
        self.print_sampling_point_info(finfo, spname)

        close_printable_file(fpressure)
        close_printable_file(fthermal)
        close_printable_file(finfo)

        for sensor in SensorType:
            senDir = create_dir(f"{labDir}/{sensor.name}")
            fname = SENSOR_FILE_NAMES[sensor]
            fp = open_printable_file(rac=senDir, fname=fname)
            self.print_sensor_file(fp, sensor.value, SENSOR_FILE_NAMES[sensor])
            close_printable_file(fp)

    def get_list_current_params(self):
        """
        Retourne les paramètres PHYSIQUES actuels de chaque couche
        en appelant la méthode de traduction.
        """
        return [layer.get_physical_params() for layer in self.all_layers]


def rmse_measured(temp_simul, temp_obs):
    """Root mean square of temp_simul - temp_obs over the measured values of temp_obs (nan if none is measured)."""
    measured = ~np.isnan(temp_obs)
    if not measured.any():
        return np.nan
    return np.sqrt(np.mean((temp_simul[measured] - temp_obs[measured]) ** 2))
