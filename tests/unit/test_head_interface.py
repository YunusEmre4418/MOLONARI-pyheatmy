"""Unit tests for the head equation at a layer interface (H_stratified.correct_numerical_schema):
the storage term of each row of the implicit (A) and explicit (B) matrices, and the time step used
by the interface rows when the time step varies."""

import numpy as np
import pytest

from pyheatmy.linear_system import H_stratified
from pyheatmy.params import calc_K

# A small two-layer column, cell centres at DZ / 2 + k * DZ
N_CELL = 6
DZ = 0.05
Z_SOLVE = DZ / 2 + DZ * np.arange(N_CELL)        # 0.025, 0.075, …, 0.275
DT = 900.0                                        # 15 min, as the Point034 data
ONE_STEP = np.array([DT])

UPPER_INTRINK = 1e-12                             # m2
LOWER_INTRINK = 1e-13
POROSITY = 0.1
LAMBDA_S = 2.0
RHOS_CS = 4e6
TEMPERATURE = 290.0

# Interface between the centres of cells 2 and 3, or on the centre of cell 3
BETWEEN_CENTRES_Z = 0.15
BETWEEN_CENTRES = np.array([[2, 3]])
ON_CENTRE_Z = Z_SOLVE[3]
ON_CENTRE = np.array([[3, 0]])

SS = np.linspace(0.2, 0.7, N_CELL)                # 1/m, a different value in every cell
DELTA_SS = 0.05
RELATIVE_TOLERANCE = 1e-9

# Variable time step: low conductivities, so that the storage term, and thus dt, weighs in the interface rows
TWO_STEPS = np.array([DT, 4 * DT])                # a gap in the series: the second step is four times longer
LOW_UPPER_INTRINK = 1e-14
LOW_LOWER_INTRINK = 1e-15
INITIAL_HEAD = np.linspace(0.3, 0.0, N_CELL)       # m
RIVER_HEAD = np.array([0.3, 0.5, 0.4])            # m, at the three times
HEAD_RTOL = 1e-12                                 # the solver works in float64

def corrected_diagonals(ss_list, z_low, inter_cara):
    """Diagonals of A and B for one time step, after the interface correction."""
    lower_cells = Z_SOLVE > z_low
    intrinK_list = np.where(lower_cells, LOWER_INTRINK, UPPER_INTRINK)
    system = H_stratified(
        Ss_list=ss_list,
        IntrinK_list=intrinK_list,
        n_list=np.full(N_CELL, POROSITY),
        lambda_s_list=np.full(N_CELL, LAMBDA_S),
        rhos_cs_list=np.full(N_CELL, RHOS_CS),
        all_dt=ONE_STEP,
        q_s_list=np.zeros(N_CELL),
        dz=DZ,
        H_init=np.zeros(N_CELL),
        H_riv=np.zeros(len(ONE_STEP) + 1),
        H_aq=np.zeros(len(ONE_STEP) + 1),
        T_init=np.full(N_CELL, TEMPERATURE),
        array_K=calc_K(np.array([UPPER_INTRINK, LOWER_INTRINK])),  # hydraulic conductivity of each layer, in m/s
        array_Ss=None,
        list_zLow=np.array([z_low]),
        z_solve=Z_SOLVE,
        inter_cara=inter_cara,
        isdtconstant=np.array(True),
    )
    lower_B, diagonal_B, upper_B = system.compute_B_diagonals(DT)
    lower_A, diagonal_A, upper_A = system.compute_A_diagonals(DT)
    system.correct_numerical_schema(0, DT, lower_B, diagonal_B, upper_B, lower_A, diagonal_A, upper_A)
    return diagonal_A, diagonal_B


CASES = [
    *(pytest.param(ON_CENTRE_Z, ON_CENTRE, cell, id=f"on_centre-cell{cell}") for cell in range(N_CELL)),
    *(pytest.param(BETWEEN_CENTRES_Z, BETWEEN_CENTRES, cell, id=f"between_centres-cell{cell}") for cell in range(N_CELL)),
]


@pytest.mark.parametrize("z_low, inter_cara, cell", CASES)
def test_storage_term_of_each_row_uses_its_own_cell(z_low, inter_cara, cell):
    """Changing Ss in one cell changes only the diagonal of that cell's row, by delta Ss / dt in A and in B:
    the time discretisation of Ss dH/dt puts Ss / dt on both diagonals."""
    changed_ss = SS.copy()
    changed_ss[cell] += DELTA_SS
    base_A, base_B = corrected_diagonals(SS, z_low, inter_cara)
    changed_A, changed_B = corrected_diagonals(changed_ss, z_low, inter_cara)

    expected = np.zeros(N_CELL)
    expected[cell] = DELTA_SS / DT
    np.testing.assert_allclose(changed_A - base_A, expected, rtol=RELATIVE_TOLERANCE, atol=RELATIVE_TOLERANCE * DELTA_SS / DT)
    np.testing.assert_allclose(changed_B - base_B, expected, rtol=RELATIVE_TOLERANCE, atol=RELATIVE_TOLERANCE * DELTA_SS / DT)



def solve_heads(all_dt, initial_head, river_head, z_low, inter_cara):
    """Heads at every time, solved by H_stratified with the low conductivities."""
    lower_cells = Z_SOLVE > z_low
    system = H_stratified(
        Ss_list=SS,
        IntrinK_list=np.where(lower_cells, LOW_LOWER_INTRINK, LOW_UPPER_INTRINK),
        n_list=np.full(N_CELL, POROSITY),
        lambda_s_list=np.full(N_CELL, LAMBDA_S),
        rhos_cs_list=np.full(N_CELL, RHOS_CS),
        all_dt=all_dt,
        q_s_list=np.zeros(N_CELL),
        dz=DZ,
        H_init=initial_head,
        H_riv=river_head,
        H_aq=np.zeros(len(all_dt) + 1),
        T_init=np.full(N_CELL, TEMPERATURE),
        array_K=calc_K(np.array([LOW_UPPER_INTRINK, LOW_LOWER_INTRINK])),
        array_Ss=None,
        list_zLow=np.array([z_low]),
        z_solve=Z_SOLVE,
        inter_cara=inter_cara,
        isdtconstant=np.all(all_dt == all_dt[0]),
    )
    return system.compute_H_stratified()


@pytest.mark.parametrize("z_low, inter_cara", [
    pytest.param(ON_CENTRE_Z, ON_CENTRE, id="on_centre"),
    pytest.param(BETWEEN_CENTRES_Z, BETWEEN_CENTRES, id="between_centres"),
])
def test_variable_time_step_run_equals_chain_of_single_steps(z_low, inter_cara):
    """A run with two different time steps gives the same heads as two runs of one step each,
    the second starting from the result of the first: each step is solved with its own dt."""
    both_steps = solve_heads(TWO_STEPS, INITIAL_HEAD, RIVER_HEAD, z_low, inter_cara)
    first = solve_heads(TWO_STEPS[:1], INITIAL_HEAD, RIVER_HEAD[:2], z_low, inter_cara)[:, 1]
    second = solve_heads(TWO_STEPS[1:], first, RIVER_HEAD[1:], z_low, inter_cara)[:, 1]
    np.testing.assert_allclose(both_steps[:, 1], first, rtol=HEAD_RTOL)
    np.testing.assert_allclose(both_steps[:, 2], second, rtol=HEAD_RTOL)
