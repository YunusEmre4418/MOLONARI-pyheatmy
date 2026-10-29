#
# NF Ce fichier est un fichier 'poubelle', on ne comprend pas à quoi sont reliées les méthodes. Réorganiser dans les .py adhoc, éventuellement en créer de nouveaux
#
from numpy import (
    float64,
    zeros,
    nansum,
    sum,
    var,
    mean,
    isclose,
    sqrt,
    all,
    array,
    shape,
    exp,
    size,
    log,
    arange,
    pi,
    sin,
    full,
    inf,
    nan,
)

# import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

from pyheatmy.config import *


def create_periodic_signal(dates : list[datetime]
,params : list,signal_name="TBD",verbose=True): #params has 3 arguments 0 --> amplitude, 1 --> period (no period for CODE_scalar), 2 --> offset
    
    # check if the time step is constant
    t_step_list = []
    for i in range(len(dates) - 1):
        t_step_list.append(dates[i+1] - dates[i])
    assert len(set(t_step_list)) <= 1, "The time step between two consecutive dates should be constant."
    
    dt = t_step_list[0].total_seconds()
    if verbose:
        print(f"Entering {signal_name} generation with amplitude {params[0]}, period of {params[1]}, offset {params[2]}, dt {dt} --> ")
    t_range = arange(len(dates)) * dt
    if params[1] != CODE_scalar :
        if verbose:
            print(f"periodic signal\n")
        signal = (params[0] * sin(2 * pi * t_range / params[1]) + params[2] )
        if verbose:
            plt.figure(figsize=(10, 5))
            plt.plot(dates, signal, label='Signal')
            # Add labels and title
            plt.xlabel('Dates')
            plt.ylabel('Signal')
            plt.title('Signal as a Function of Dates')
            plt.legend()
            # Rotate date labels for better readability
            plt.xticks(rotation=45)
            # Display the plot
            plt.tight_layout()
            plt.show()
    else:
        if verbose:
            print(f"constant signal\n")
        signal = full(len(dates), params[2])
    return signal    

# Modification to handle multiple periodic signals
def create_multi_periodic_signal(
    dates: list[datetime],
    params,
    signal_name="TBD",
    verbose=True,
):
    """
    Build a signal from one or many sets of periodic parameters.

    Parameters
    ----------
    dates : list[datetime]
        Time axis used for evaluation. A constant spacing is required.
    params : list or list[list]
        Either a single periodic definition [amplitude, period, offset] or
        an iterable of such definitions.
    """
    if len(dates) < 2:
        raise ValueError("You need to specify an offset and at least one signal component.")

    # Build time step list to ensure the sampling is regular.
    t_step_list = [dates[i + 1] - dates[i] for i in range(len(dates) - 1)]
    assert (
        len(set(t_step_list)) <= 1
    ), "The time step between two consecutive dates should be constant."

    # Require params to be a sequence of lists/tuples in the new format.
    # New required formats:
    #   [[offset], [amp, period, phase], ...]
    #   [[amp, period, phase], ...]  (no global offset)
    if not isinstance(params, (list, tuple)):
        raise TypeError("params must be a list of lists. See function docstring for expected format.")
    if not params:
        raise ValueError("params must contain at least one periodic definition.")

    # First element can be a single-element list -> global offset
    if isinstance(params[0], (list, tuple)) and len(params[0]) == 1:
        global_offset = params[0][0]
        components = list(params[1:])
        if len(components) == 0:
            # Only an offset provided
            if verbose:
                print("Only global offset provided. Returning constant signal.")
            return full(len(dates), global_offset, dtype=float64)
    else:
        # No global offset supplied; treat all entries as component triplets
        global_offset = 0.0
        components = list(params)

    # Validate that every component is a triplet [amplitude, period, phase]
    for idx, comp in enumerate(components):
        if not isinstance(comp, (list, tuple)) or len(comp) != 3:
            raise ValueError(f"Component at index {idx} must be a triplet [amplitude, period, phase].")

    dt = t_step_list[0].total_seconds()
    t_range = arange(len(dates)) * dt
    signal = zeros(len(dates), dtype=float64)

    if verbose:
        print("Multiple periodic signals detected, summing components with global offset.")

    for idx, param in enumerate(components):
        if not isinstance(param, (list, tuple)) or len(param) != 3:
            raise ValueError(
                f"Component at index {idx} should be a triplet [amplitude, period, phase]."
            )

        amplitude, period, phase = param
        # If period is CODE_scalar treat as a constant component. We use amplitude + phase as the value
        # for backward-like behaviour where a constant component previously used the 'offset' slot.
        if period == CODE_scalar:
            component = full(len(dates), amplitude + phase, dtype=float64)
        else:
            component = amplitude * sin(2 * pi * t_range / period + phase)
        signal += component

    # Add global offset
    if global_offset != 0:
        signal = signal + global_offset

    if verbose:
        plt.figure(figsize=(10, 5))
        plt.plot(dates, signal, label="Signal")
        plt.xlabel("Dates")
        plt.ylabel("Signal")
        plt.title(f"{signal_name} as a Function of Dates")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    return signal


def create_dir(rac, verbose=True):
    # Directory path to print in
    dir_print = os.path.expanduser(rac)
    # Create the folder and subfolder
    os.makedirs(dir_print, exist_ok=True)
    return dir_print


def open_printable_file(
    rac,
    dataType=None,
    classType=None,
    verbose=True,
    fname=None,
    spname=None,
    ext=".csv",
):
    dir_print = rac
    if dataType != None and classType != None:
        dataname = DEVICE_FILE_NAMES[dataType]
        origin = CLASS_FILE_NAMES[classType]
        fname = f"{dir_print}/{spname}_{dataname}_{origin}{ext}"
        if verbose:
            print(f"Creating {fname}")

        fp = open(fname, "w")
        if dataType == DeviceType.PRESSURE:
            fp.write(
                "“Date/heure”, “Hydraulic head differential in m, “Temperature in °C””\n"
            )
        else:
            fp.write(
                "“time”,”T°C sensor 1”,”T°C sensor 2”,”T°C sensor 3”,”T°C sensor 4”\n"
            )
    else:
        if fname != None:
            fname = f"{dir_print}/{fname}{ext}"
        else:
            fname = f"{dir_print}/pyheatmy_default{ext}"
        if verbose:
            print(f"Creating {fname}")
        fp = open(fname, "w")

    return fp


def close_printable_file(fp, verbose=True):
    if not fp.closed:
        fp.close()
        if verbose:
            print(f"File {fp.name} closed successfully.")
    else:
        if verbose:
            print(f"File {fp.name} is already closed.")

