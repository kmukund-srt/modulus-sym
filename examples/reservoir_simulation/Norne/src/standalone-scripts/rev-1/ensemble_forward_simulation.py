from util_functions import *
import os
from os.path import abspath
# from NVRS import *
import numpy as np
from typing import Dict
from loky import get_reusable_executor
from joblib import Parallel, delayed
import shutil

# base_dir is the name of the base directory to copy all the simulation files
base_dir = "/workspace/project/src/standalone-scripts/rev-1/test1/base-case/"
base_filename = "FULLNORNE"
runsim = "echelon --ngpus=1 " + base_filename + ".DATA"

# root name of the directory where all the ensemble simulations are stored
ensemble_dir_root_name = "/workspace/project/src/standalone-scripts/rev-1/test1/ensemble/"
os.makedirs(ensemble_dir_root_name, exist_ok=True)

# base directory name for ensemble members -- the ensemble member number will be
# appended. For example, Realization0, Realization1, ... Realization100 etc.
realization_prefix_dirname = "Realization"

# Paths for input data for priors (permeability, porosity, faults)
perm_file_path = base_dir + "sgsim.out"
poro_file_path = base_dir + "sgsimporo.out"
fault_file_path = base_dir + "faultensemble.dat"

# size of the ensemble
N_ens = 100

# Number of parallel jobs to be launched.
N_parallel = 10

USE_PARALLEL_LAUNCHING = False


def copy_files(base_dir, dest_dir):
    files = os.listdir(base_dir)
    for file in files:
        shutil.copy(os.path.join(base_dir, file), dest_dir)


def save_files(perm, poro, perm2, dest_dir, oldfolder):
    os.chdir(dest_dir)
    filename1 = "permx" + ".dat"
    np.savetxt(
        filename1,
        perm,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PERMX",
        footer="/",
        comments="",
    )

    filename2 = "porosity" + ".dat"
    np.savetxt(
        filename2,
        poro,
        fmt="%.4f",
        delimiter=" \t",
        newline="\n",
        header="PORO",
        footer="/",
        comments="",
    )

    my_array = perm2.ravel()
    my_array_index = 0

    # read the file
    with open("multflt.dat", "r") as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        if line.strip() == "MULTFLT":
            # do nothing if this is the MULTFLT line
            continue
        else:
            # replace numerical values on other lines
            parts = line.split(" ")
            if (
                len(parts) > 1
                and parts[1].replace(".", "", 1).replace("/", "").isdigit()
            ):
                parts[1] = str(my_array[my_array_index])
                lines[i] = " ".join(parts)
                my_array_index += 1

    # write the file
    with open("multflt.dat", "w") as file:
        file.writelines(lines)

    os.chdir(oldfolder)


def run_simulator(dest_dir, oldfolder):
    os.chdir(dest_dir)
    os.system(runsim)
    os.chdir(oldfolder)


def run_ensemble(N_ens: int) -> None:
    oldfolder = os.getcwd()
    os.chdir(oldfolder)
    njobs = min(N_parallel, N_ens)

    perm_ensemble = np.genfromtxt(abspath(perm_file_path))
    poro_ensemble = np.genfromtxt(abspath(poro_file_path))
    fault_ensemble = np.genfromtxt(abspath(fault_file_path))

    perm_ensemble = clip_and_convert_to_float32(perm_ensemble)
    poro_ensemble = clip_and_convert_to_float32(poro_ensemble)
    fault_ensemble = clip_and_convert_to_float32(fault_ensemble)

    ensembles_root_name = ensemble_dir_root_name + realization_prefix_dirname
    for kk in range(N_ens):
        print("------------ Simulation number: ", kk, " ----------------")
        path_out = abspath(ensembles_root_name + str(kk))
        os.makedirs(path_out, exist_ok=True)

    executor_copy = get_reusable_executor(max_workers=njobs)
    with executor_copy:
        Parallel(n_jobs=njobs, backend="multiprocessing", verbose=50)(
            delayed(copy_files)(
                base_dir, abspath(ensembles_root_name + str(kk))
            )
            for kk in range(N_ens)
        )
        executor_copy.shutdown(wait=False)

    executor_save = get_reusable_executor(max_workers=njobs)
    with executor_save:
        Parallel(n_jobs=njobs, backend="multiprocessing", verbose=50)(
            delayed(save_files)(
                perm_ensemble[:, kk],
                poro_ensemble[:, kk],
                fault_ensemble[:, kk],
                abspath(ensembles_root_name + str(kk)),
                oldfolder,
            )
            for kk in range(N_ens)
        )
        executor_save.shutdown(wait=False)

    print("")
    print("---------------------------------------------------------------------")
    print("")
    print("\n")
    print("|-----------------------------------------------------------------|")
    print("|                 RUN FLOW SIMULATOR FOR ENSEMBLE                  |")
    print("|-----------------------------------------------------------------|")
    print("")

    if USE_PARALLEL_LAUNCHING == True:
        executor_run = get_reusable_executor(max_workers=njobs)
        with executor_run:
            Parallel(n_jobs=njobs, backend="multiprocessing", verbose=50)(
                delayed(run_simulator)(
                    abspath(ensembles_root_name + str(kk)), oldfolder)
                for kk in range(N_ens)
            )
            executor_run.shutdown(wait=False)
    else:
        for kk in range(N_ens):
            print("|-----------------------------------------------------------------|")
            print("|                 EXECUTING SIMULATION:",
                  kk, "                  |")
            print("|-----------------------------------------------------------------|")
            run_simulator(abspath(ensembles_root_name + str(kk)), oldfolder)

    print("|-----------------------------------------------------------------|")
    print("|                 EXECUTED RUNS of FLOW SIMULATION FOR ENSEMBLE   |")
    print("|-----------------------------------------------------------------|")


if __name__ == "__main__":
    # start_mps()
    run_ensemble(N_ens)
