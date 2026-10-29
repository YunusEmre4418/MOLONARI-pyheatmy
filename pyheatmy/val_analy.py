import numpy as np
from typing import Sequence

from pyheatmy.core import Column
from pyheatmy.synthetic_MOLONARI import synthetic_MOLONARI
from pyheatmy.layers import Layer
from pyheatmy.params import calc_K
from pyheatmy.config import *

class Analy_Sol:
    "Compute the value of temperature based on the analytical solution."
    def __init__( 
        self,
        column_exp : Column,
        time_series : synthetic_MOLONARI,
        monolayer : Layer,
        nb_cells : int,
    ):
        self._depth_sensors = column_exp.depth_sensors
        self._real_z_sensors = column_exp._real_z

        self._nb_cells = nb_cells
        self._dz = self._real_z_sensors[-1] / self._nb_cells
        self._z_solve = self._dz/2 + np.array([k*self._dz for k in range(self._nb_cells)])
        self._id_sensors = [np.argmin(np.abs(z - self._z_solve)) for z in column_exp._real_z[1:-1]]

        self._real_t = time_series._time_array
        # La solution analytique suppose une charge constante [[dH]], une température de rivière sinusoïdale
        # [[T_moy], [amplitude, période, phase]] et une température d'aquifère constante égale à T_moy.
        try:
            [[self._dH]] = time_series._param_dH
            [[self._T_moy], [self._amp_T_river, self._period, self._phase]] = time_series._param_T_riv
            [[T_aq]] = time_series._param_T_aq
        except (TypeError, ValueError):
            raise ValueError(
                "Analy_Sol needs param_dH_signal = [[dH]], param_T_riv_signal = [[T_mean], [amplitude, period, phase]] "
                "and param_T_aq_signal = [[T_mean]]."
            ) from None
        if T_aq != self._T_moy:
            raise ValueError(f"Analy_Sol needs the same mean temperature in the river and the aquifer, got {self._T_moy} and {T_aq}.")

        IntrinK, self._n, self._lambda_s, self._rho_cs, q_s = monolayer.get_physical_params()
        if q_s != 0:
            raise ValueError(f"Analy_Sol has no source term, got q_s = {q_s}.")
        self._K = calc_K(IntrinK)

        self._lambda_m = (self._n * (LAMBDA_W) ** 0.5 + (1.0 - self._n) * (self._lambda_s) ** 0.5) ** 2
        self._rho_cm = self._n * RHO_W * C_W + (1 - self._n) * self._rho_cs
        
        self._alpha = (RHO_W*C_W/self._rho_cm)*self._K
        self._kappa = self._lambda_m/self._rho_cm

        # Vitesse du front thermique : flux de Darcy K dH / L sur la hauteur L de la colonne, positif vers le bas
        # (selon z), comme dans le modèle direct.
        self._v_t = self._alpha*self._dH/self._real_z_sensors[-1]
        self._a_cond = np.sqrt(np.pi/(self._kappa*self._period))

        self._a = None
        self._b = None

        self.analy_temp_general = np.array([None])
        self.analy_temp_cond = np.array([None])

    @classmethod
    def from_dict(cls, analy_sol_dict):
        return cls(**analy_sol_dict)
    
    def compute_a(self):
        self._a = (np.sqrt((np.sqrt(self._v_t**4 + (8*np.pi*self._kappa/self._period)**2) + self._v_t**2)/2) - self._v_t)/(2*self._kappa)

    def compute_b(self): 
        self._b = np.sqrt((np.sqrt(self._v_t**4 + (8*np.pi*self._kappa/self._period)**2) - self._v_t**2)/2)/(2*self._kappa)

    def compute_temp_general(self):
        # on maj les valeurs de a et b
        self.compute_a()
        self.compute_b()
        self.analy_temp_general = np.zeros((self._nb_cells,len(self._real_t)))
        for i, t in enumerate(self._real_t) :      
            self.analy_temp_general[:,i] = self._T_moy + self._amp_T_river*np.exp(-self._a*self._z_solve)*np.sin(2*np.pi*t/self._period + self._phase - self._b*self._z_solve)


    def compute_temp_cond(self):
        self.analy_temp_cond = np.zeros((self._nb_cells,len(self._real_t)))
        for i, t in enumerate(self._real_t) : 
            self.analy_temp_cond[:,i] = self._T_moy + self._amp_T_river*np.exp(-self._a_cond*self._z_solve)*np.sin(2*np.pi*t/self._period + self._phase - self._a_cond*self._z_solve)

            
    def generate_RMSE_analytical(self, time_series, column, monolayer): #column défini à partir de time_series
        """This function computes the RMSE of the direct model compared to analytical solutions on simple boundary
        conditions and in a monolayer case: one value per sensor, then the RMSE over all sensors.
        The direct model starts from the initial profile of the column, built from its measurements at the first
        time: for a column that starts from the analytical solution, give it these values when creating it."""
        if self._a is None : # compute_temp_general n'a pas encore été appelée
            self.compute_temp_general()

        column.set_layers(monolayer)
        time_series._measures_column_one_layer(column)

        self.temp_shaft_analy = self.analy_temp_general[self._id_sensors, :].T #ne contient pas la boundary condition aquifer
        temp_shaft_direct = time_series._T_Shaft[:, :-1]
        # Array of RMSE for each sensor
        list_RMSE = np.sqrt(np.mean((self.temp_shaft_analy - temp_shaft_direct)**2, axis=0))

        # Total RMSE
        total_RMSE = np.sqrt(np.mean(list_RMSE**2))

        return np.append(list_RMSE, total_RMSE)