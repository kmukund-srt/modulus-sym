from util_functions import *
from EchOutputParser import *
import os
from os.path import abspath, dirname
import numpy as np
from typing import Dict
import gc
import shutil
from shutil import rmtree
from importlib import reload
from skimage.transform import resize
import scipy.io as scipy_io

import torch
from torch import device

# Profiling tools
import cProfile
from cProfile import Profile
from pstats import SortKey, Stats

import EchOutputParser
reload(EchOutputParser)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

DELETE_DIRECTORIES = False
USE_FAULTS = False
PROFILE_CODE = False

""" Initialize params. """
Nens = 100
base_dir = "/workspace/project/src/standalone-scripts/rev-1/test1/base-case/"
ensemble_dir_root_name = "/workspace/project/src/standalone-scripts/rev-1/test1/ensemble/"
realization_prefix_dirname = "Realization"
base_filename = "FULLNORNE"

simulation_train_wells_data_file_path = abspath(
    ensemble_dir_root_name + "PACKETS/data_train_peaceman.pkl.gz")
simulation_test_wells_data_file_path = abspath(
    ensemble_dir_root_name + "PACKETS/data_test_peaceman.pkl.gz")
conversions_file_path = abspath(
    ensemble_dir_root_name + "PACKETS/conversions.mat")
simulation_train_data_file_path = abspath(
    ensemble_dir_root_name + "PACKETS/data_train.pkl.gz")
simulation_test_data_file_path = abspath(ensemble_dir_root_name +
                                         "PACKETS/data_test.pkl.gz")

time_slices_to_include = np.linspace(1, 246, 10, dtype=int)-1  # 0-indexing

STATE_VARS = ('PRESSURE', 'SGAS', 'SWAT', 'RS', 'RV')
PROPS_VARS = ('PORO', 'PERMX', 'PERMY', 'PERMZ')

prod_wells_list = [
    "B-1BH",
    "B-1H",
    "B-2H",
    "B-3H",
    "B-4BH",
    "B-4DH",
    "B-4H",
    "D-1CH",
    "D-1H",
    "D-2H",
    "D-3AH",
    "D-3BH",
    "D-4AH",
    "D-4H",
    "E-1H",
    "E-2AH",
    "E-2H",
    "E-3AH",
    "E-3CH",
    "E-3H",
    "E-4AH",
    "K-3H",
]

watinj_wells_list = {
    "C-1H",
    "C-2H",
    "C-3H",
    "C-4AH",
    "C-4H",
    "F-1H",
    "F-2H",
    "F-3H",
    "F-4H",
}

gasinj_wells_list = {
    "C-1H",
    "C-3H",
    "C-4AH",
    "C-4H",
}


def get_faults_data(
    path_to_result,
    nx,
    ny,
    nz,
):
    os.chdir(path_to_result)

    """
    This section is for the fault multiplier values
    """
    float_parameters = []
    file_path = "multflt.dat"
    with open(file_path, "r") as file:
        for line in file:
            # Use split to separate the string into parts.
            split_line = line.split()

            # Ensure there are at least two elements in the split line
            if len(split_line) >= 2:
                try:
                    # The second element (index 1) should be the float you're interested in.
                    float_parameter = float(split_line[1])
                    float_parameters.append(float_parameter)
                except ValueError:
                    # Handle cases where the second element can't be converted to a float.
                    pass
            else:
                pass
    floatts = np.hstack(float_parameters)

    Fault = np.ones((nx, ny, nz), dtype=np.float16)
    Fault[8:14, 61:81, 0:] = floatts[0]  # E_01
    Fault[7, 56:61, 0:] = floatts[1]  # E_01_F3
    Fault[9:16, 53:69, 0:15] = floatts[2]  # DE_1
    Fault[9:16, 53:69, 15:22] = floatts[3]  # DE_1LTo
    Fault[8:10, 48:53, 0:] = floatts[4]  # DE_B3
    Fault[5:8, 46:93, 0:] = floatts[5]  # DE_0
    Fault[15, 69:100, 0:] = floatts[6]  # DE_2
    Fault[5:46, 7:10, 0:11] = floatts[7]  # BC
    Fault[10:16, 44:52, 0:11] = floatts[8]  # CD
    Fault[10:16, 44:52, 11:22] = floatts[9]  # CD_To
    Fault[6:11, 38:44, 0:] = floatts[10]  # CD_B3
    Fault[5:7, 38, 0:] = floatts[11]  # CD_0
    Fault[15:19, 52:54, 0:] = floatts[12]  # CD_1
    Fault[26, 13:49, 0:] = floatts[13]  # C_01
    Fault[26, 42:48, 0:] = floatts[14]  # C_01_Ti
    Fault[24, 45:52, 10:19] = floatts[15]  # C_08
    Fault[24, 45:52, 0:10] = floatts[16]  # C_08_Ile
    Fault[24, 40:45, 0:19] = floatts[17]  # C_08_S
    Fault[24, 45:52, 19:22] = floatts[18]  # C_08_Ti
    Fault[24, 40:45, 19:22] = floatts[19]  # C_08_S_Ti
    Fault[21:25, 48:68, 0:] = floatts[20]  # C_09
    Fault[5:7, 18:22, 0:] = floatts[21]  # C_02
    Fault[23:24, 25:35, 0:] = floatts[22]  # C_04
    Fault[12:14, 41:44, 0:] = floatts[23]  # C_05
    Fault[16:19, 47:52, 0:] = floatts[24]  # C_06
    Fault[6:18, 20:22, 0:] = floatts[25]  # C_10
    Fault[25, 51:57, 0:] = floatts[26]  # C_12
    Fault[21:24, 35:39, 0:15] = floatts[27]  # C_20
    Fault[21:24, 35:39, 15:22] = floatts[28]  # C_20_LTo
    Fault[11:13, 12:24, 0:17] = floatts[29]  # C_21
    Fault[11:13, 12:24, 17:22] = floatts[30]  # C_21_Ti
    Fault[12:14, 14:18, 0:] = floatts[31]  # C_22
    Fault[19:22, 19, 0:] = floatts[32]  # C_23
    Fault[16:18, 25:27, 0:] = floatts[33]  # C_24
    Fault[20:22, 24:30, 0:] = floatts[34]  # C_25
    Fault[7:9, 27:36, 0:] = floatts[35]  # C_26
    Fault[6:8, 36:44, 0:] = floatts[36]  # C_26N
    Fault[9, 29:38, 0:] = floatts[37]  # C_27
    Fault[14:16, 38:40, 0:] = floatts[38]  # C_28
    Fault[14:16, 43:46, 0:] = floatts[39]  # C_29
    Fault[18, 47:88, 0:] = floatts[40]  # DI
    Fault[18, 44:47, 0:] = floatts[41]  # DI_S
    Fault[16:18, 55:65, 0:] = floatts[43]  # D_05
    Fault[6:10, 78:92, 0:] = floatts[43]  # EF
    Fault[28:38, 53:103, 0:] = floatts[44]  # GH
    Fault[30:33, 71:86, 0:] = floatts[45]  # G_01
    Fault[34:36, 72:75, 0:] = floatts[46]  # G_02
    Fault[33:37, 70, 0:] = floatts[47]  # G_03
    Fault[30:34, 61:65, 0:] = floatts[48]  # G_05
    Fault[30:37, 79:92, 0:] = floatts[49]  # G_07
    Fault[34, 104:107, 0:] = floatts[50]  # G_08
    Fault[30:37, 66:70, 0:] = floatts[51]  # G_09
    Fault[35:40, 64:69, 0:] = floatts[52]  # G_13

    return Fault


def get_grid_dims(grid_data):
    nx, ny, nz = grid_data['GRID DATA']['GRIDHEAD'][0][1:4]
    return nx, ny, nz


def get_active_index_array_full_grid(grid_data):
    # ACTNUM is the keyword denoting active cells
    active_cells_mask_array = grid_data['GRID DATA']['ACTNUM'][0]
    # ACTNUM = 1 means the cell is active, 0 means inactive
    active_index_array = np.where(active_cells_mask_array == 1)[0]
    return active_index_array


def get_time_slices(sim_output):
    all_report_times = sim_output['REPORT TIMES']
    report_time_slices = all_report_times if time_slices_to_include is None else all_report_times[
        time_slices_to_include]
    return report_time_slices


def gets_wells_and_indices(sim_output, well_type: str, wells_list=None) -> Dict:
    ###################################################################
    # Get the I,J,K indices of the wells.
    # Because of the existing Modulus code that assumes
    # all wells and all connections are open all the time
    # we pick all the producing wells from the last time index frame
    # during which we know for Norne, all wells are active
    ###################################################################
    wells_conns_type_status_dict = sim_output.get('WELLS AND CONNECTIONS', {})

    if wells_list is None:
        # Return the list of all wells of type well_type
        wells_subset_all_times = {time_index: {well_name: details for well_name, details in wells_dict.items()
                                               if details.get('TYPE') == well_type}
                                  for time_index, wells_dict in wells_conns_type_status_dict.items()}
    else:
        # Return wells based on its type and also in wells_list.
        # This joint filter is necessary because, for example, Norne has 27 producing wells and
        # only a subset of 22 are chosen (refer, Clement Etienam)
        wells_subset_all_times = {time_index: {well_name: details for well_name, details in wells_dict.items()
                                               if details.get('TYPE') == well_type and well_name in wells_list}
                                  for time_index, wells_dict in wells_conns_type_status_dict.items()}

    # Important Note:
    #      Only the last time slice is considered assuming that it has information about
    #      all the wells in the simulation. Wells coe online at different time periods
    #      and so this was considered a safer choice.
    report_time_slices = get_time_slices(sim_output)
    last_time_slice = len(report_time_slices)-1
    wells_IJK_dict = {well_name: details['Connections']['Indices']
                      for well_name, details in wells_subset_all_times[last_time_slice].items()}

    return wells_IJK_dict


def get_solution_summary_data(sim_output, wells_list):
    wells_solution_summary_data = sim_output['WELLS SOLUTION SUMMARY']
    wells_solution_summary_data = {smry: {well_name: qty for well_name, qty in details.items()
                                          if well_name in wells_list}
                                   for smry, details in wells_solution_summary_data.items()}
    return wells_solution_summary_data


def get_injector_data(grid_data, sim_output):
    watinj_wells_data = get_solution_summary_data(
        sim_output, watinj_wells_list)
    gasinj_wells_data = get_solution_summary_data(
        sim_output, gasinj_wells_list)

    nx, ny, nz = get_grid_dims(grid_data)
    report_time_slices = get_time_slices(sim_output)
    len_times = len(report_time_slices)
    # 1 probably corresponds to ensemble size
    Qw = np.zeros((1, len_times, nx, ny, nz), dtype=np.float32)
    Qg = np.zeros((1, len_times, nx, ny, nz), dtype=np.float32)
    Qo = np.zeros((1, len_times, nx, ny, nz), dtype=np.float32)

    QW = np.zeros((len_times, nx, ny, nz), dtype=np.float32)
    QG = np.zeros((len_times, nx, ny, nz), dtype=np.float32)
    QO = np.zeros((len_times, nx, ny, nz), dtype=np.float32)

    watinj_wells_IJK_dict = gets_wells_and_indices(
        sim_output, 'WATER INJECTOR', watinj_wells_list)
    gasinj_wells_IJK_dict = gets_wells_and_indices(
        sim_output, 'GAS INJECTOR', gasinj_wells_list)
    producer_wells_IJK_dict = gets_wells_and_indices(
        sim_output, 'PRODUCER', prod_wells_list)

    # We need to scatter the time series data to the full grid
    # for the injector wells at their respective completion indices.
    # All the completion indices for a given well will be populated with
    # the same value for a given time slice.
    for well, conn_indices in watinj_wells_IJK_dict.items():
        # Transpose and unpack the indices for advanced indexing (See below for np.mean)
        i, j, k = conn_indices.T
        # Note: watinj_wells_data is used starting with idx = 1. First data at t=0 is neglected
        for idx in range(len_times):
            QW[idx, i, j, k] = watinj_wells_data['WWIR'][well][idx]

    for well, conn_indices in gasinj_wells_IJK_dict.items():
        # Transpose and unpack the indices for advanced indexing (See below for np.mean)
        i, j, k = conn_indices.T
        # Note: gasinj_wells_data is used starting with idx = 1. First data at t=0 is neglected
        for idx in range(len_times):
            QG[idx, i, j, k] = gasinj_wells_data['WGIR'][well][idx]

    for well, conn_indices in producer_wells_IJK_dict.items():
        # Transpose and unpack the indices for advanced indexing (See below for np.mean)
        i, j, k = conn_indices.T
        # Note: For some reason, all the producer connections are populated with -1.
        QO[:, i, j, k] = -1

    '''

    for well, conn_indices in gasinj_wells_IJK_dict.items():
        i,j,k = conn_indices.T # Transpose and unpack the indices for advanced indexing (See below for np.mean)
        # Ensure WGIR[well] is broadcastable to the shape defined by i, j, k for each time slice
        # This code assumes that gasinj_wells_data['WGIR'][well] is a 1D array with a length that matches 
        # the number of time slices in QG. The [:, None, None, None] part is used to add 
        # singleton dimensions for broadcasting to work with the advanced indexing on the 
        # left-hand side.
        QG[:,i,j,k] = gasinj_wells_data['WGIR'][well][:, None, None, None]

    for well, conn_indices in producer_wells_IJK_dict.items():
        i,j,k = conn_indices.T # Transpose and unpack the indices for advanced indexing (See below for np.mean)
        # For some reason, all the producer connections are populated with -1. 
        QO[:,i,j,k] = -1

    '''
    Qw[0, :, :, :, :] = QW
    Qg[0, :, :, :, :] = QG
    Qo[0, :, :, :, :] = QO

    return Qw, Qg, Qo


def get_prod_wells_output_fnn(sim_output):
    """Get NORNE-model specific well data.

    Parameters
    ----------

    Returns
    -------
    """
    prod_wells_data = get_solution_summary_data(sim_output, prod_wells_list)
    report_time_slices = get_time_slices(sim_output)
    num_time_slices = len(report_time_slices)

    WOPR = np.zeros((num_time_slices, len(prod_wells_data['WOPR'])))
    WWPR = np.zeros((num_time_slices, len(prod_wells_data['WWPR'])))
    WGPR = np.zeros((num_time_slices, len(prod_wells_data['WGPR'])))

    # Since the producer wells are the same for all phases, we use a single
    # loop to populate the out array for the NN. We sort the wells for this
    # loop to be consistent with the input data for the NN.
    # Refer to get_prod_wells_input_fnn function below where the wells are sorted
    for idx, well_name in enumerate(sorted(prod_wells_data['WOPR'])):
        WOPR[:, idx] = prod_wells_data['WOPR'][well_name]
        WWPR[:, idx] = prod_wells_data['WWPR'][well_name]
        WGPR[:, idx] = prod_wells_data['WGPR'][well_name]

    return np.hstack((WOPR, WWPR, WGPR))


def get_prod_wells_input_fnn(sim_output, grid_state_output, permeability):
    """Get NORNE-model specific well data.

    Parameters
    ----------

    Returns
    -------
    """
    pressure = np.array(grid_state_output['PRESSURE'])
    Swater = np.array(grid_state_output['SWAT'])
    Sgas = np.array(grid_state_output['SGAS'])
    report_time_slices = get_time_slices(sim_output)

    if PROFILE_CODE:
        print(f"---------- Profiling Listcomp: producer_wells_IJK_dict --------------")
        with Profile() as profile:
            producer_wells_IJK_dict = gets_wells_and_indices(
                sim_output, 'PRODUCER', prod_wells_list)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        producer_wells_IJK_dict = gets_wells_and_indices(
            sim_output, 'PRODUCER', prod_wells_list)

    num_time_slices = len(report_time_slices)
    num_prod_wells = len(producer_wells_IJK_dict)

    # Compute the K-averaged permeabilities for each well
    # The values for each well don't change with time and so
    # are same for every time (row).
    # perm = permeability[0, :, :, :]
    permxx_avg = np.ones((num_time_slices, num_prod_wells))
    for idx, (well, conn_indices) in enumerate(producer_wells_IJK_dict.items()):
        # Transpose and unpack the indices for advanced indexing (See below for np.mean)
        i, j, k = conn_indices.T
        permxx_avg[:, idx] *= np.mean(permeability[i, j, k])

    field_pressure = np.zeros((num_time_slices, 1))
    sgas_Kdir_avg = np.zeros((num_time_slices, num_prod_wells))
    swat_Kdir_avg = np.zeros((num_time_slices, num_prod_wells))
    time = np.zeros((num_time_slices, 1))
    for time_idx in range(num_time_slices):
        # Entire field (nx * ny * nz cells) average pressure is computed for
        # every time slice and stored
        field_pressure[time_idx, 0] = np.mean(pressure[time_idx, :, :, :])

        # Loop over all wells at a given time (k)
        gas_sat = Sgas[time_idx, :, :, :]
        wat_sat = Swater[time_idx, :, :, :]
        for well_idx, (well_name, conn_indices) in enumerate(producer_wells_IJK_dict.items()):
            # Transpose and unpack the indices for advanced indexing (See below for np.mean)
            i, j, k = conn_indices.T
            sgas_Kdir_avg[time_idx, well_idx] = np.mean(gas_sat[i, j, k])
            swat_Kdir_avg[time_idx, well_idx] = np.mean(wat_sat[i, j, k])

        time[time_idx, 0] = report_time_slices[time_idx]

    return np.hstack((permxx_avg, field_pressure,
                      1 - (sgas_Kdir_avg +
                           swat_Kdir_avg), sgas_Kdir_avg, swat_Kdir_avg,
                      time))


def scatter_arrays_to_full_grid(grid_data, init_data, sim_output):
    """ Given arrays of state variables and properties sized only to active cells length, 
    scatter them to the full grid that also includes inactive cells.

    Parameters
    ----------

    Returns
    -------
    """

    # Initialize lists to store resized slices.
    # All state variables need to be resized and reshaped to the DIMENS values!!!
    active_index_array = get_active_index_array_full_grid(grid_data)
    len_act_indx = len(active_index_array)
    nx, ny, nz = get_grid_dims(grid_data)
    resized_states_dict = {}
    resized_props_dict = {}

    def resize_fn(input_array, active_index_array):
        # Initialize an array of zeros
        resized_array = np.zeros((nx * ny * nz, 1))
        # Resize and update the array at active indices
        resized_array[active_index_array] = resize(
            input_array.reshape(-1, 1), (len_act_indx, ), order=1, preserve_range=True
        )
        # Reshape to 3D grid
        resized_array = np.reshape(resized_array, (nx, ny, nz), "F")
        return resized_array

    for var in STATE_VARS:
        for slice in sim_output['STATES'][var]:
            resized_state = resize_fn(slice, active_index_array)
            resized_states_dict.setdefault(var, []).append(resized_state)

    for var in PROPS_VARS:
        for slice in init_data['INITIAL DATA'][var]:
            resized_props = resize_fn(slice, active_index_array)
            resized_props_dict.setdefault(var, []).append(resized_props)

    return resized_states_dict, resized_props_dict


def get_initial_pressure_saturation(maxP, nx, ny, nz, Nens):
    pini_alt = 600
    sat_ini = 0.2
    ini_pressure = pini_alt * np.ones((Nens, 1, nx, ny, nz), dtype=np.float32)
    ini_pressure = ini_pressure / maxP
    ini_pressure = clip_and_convert_to_float32(ini_pressure)

    ini_sat = sat_ini * np.ones((Nens, 1, nx, ny, nz), dtype=np.float32)
    return ini_pressure, ini_sat


def generate_data(ensemble_dir_root_name, realization_prefix_dirname, base_filename,  N_ens: int) -> None:
    print("|-----------------------------------------------------------------|")
    print("|                 DATA CURATION IN PROCESS                        |")
    print("|-----------------------------------------------------------------|")

    pressure = []
    Sgas = []
    Swater = []
    Fault = []
    ReportTimes = []
    inn_fcn = []
    out_fcn = []

    # Instantiate the Norne model parser class
    # norne_model = NorneModel()
    ech_bin_parser = EchBinaryOutputParser()

    # Get the model dimensions from any ensemble realization.
    # Here, I have chosen ../ECHRUNS/Realisation0
    folder_prefix = ensemble_dir_root_name + realization_prefix_dirname
    folder = abspath(folder_prefix + str(0))
    print("-------- Folder : ", folder)

    grid_data = ech_bin_parser.get_grid_data(folder, base_filename)
    nx, ny, nz = get_grid_dims(grid_data)

    # Initialize permeability, porosity and actnum arrays
    permeability = np.zeros((N_ens, 1, nx, ny, nz))
    porosity = np.zeros((N_ens, 1, nx, ny, nz))
    actnumm = np.zeros((N_ens, 1, nx, ny, nz))

    for i in range(N_ens):
        folder = abspath(folder_prefix + str(i))
        print("|-----------------------------------------------------------------|")
        print("|                 PROCESSING ENSEMBLE MEMBER", i, "/", N_ens, "  |")
        print("|-----------------------------------------------------------------|")

        if PROFILE_CODE:
            print(
                f"---------- Profiling ech_bin_parser.get_simulation_data --------------")
            with Profile() as profile:
                sim_output = ech_bin_parser.get_simulation_data(
                    folder, base_filename, time_slices_to_include)
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.CALLS)
                    .print_stats()
                )
        else:
            sim_output = ech_bin_parser.get_simulation_data(
                folder, base_filename, time_slices_to_include)

        grid_data = ech_bin_parser.get_grid_data(folder, base_filename)
        init_data = ech_bin_parser.get_init_data(folder, base_filename)

        # Scatter the time values to the full grid
        # This is NOT the right thing to do but is currently done to
        # perform timestepping of state variables on a per-cell basis.
        time_values_on_grid_cells = []
        report_time_slices = get_time_slices(sim_output)
        for time in report_time_slices:
            time_values_on_grid_cells.append(
                time*np.ones((nx, ny, nz), dtype=np.float32))

        if USE_FAULTS:
            flt = get_faults_data(folder, nx, ny, nz)
        else:
            flt = np.ones((nx, ny, nz), dtype=np.float16)

        perm = permeability[i, 0, :, :, :]

        if PROFILE_CODE:
            print(f"---------- Profiling scatter_arrays_to_full_grid --------------")
            with Profile() as profile:
                resized_states_dict, resized_props_dict = scatter_arrays_to_full_grid(
                    grid_data, init_data, sim_output)
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.CALLS)
                    .print_stats()
                )
        else:
            resized_states_dict, resized_props_dict = scatter_arrays_to_full_grid(
                grid_data, init_data, sim_output)

        if PROFILE_CODE:
            print(f"---------- Profiling get_prod_wells_input_fnn --------------")
            with Profile() as profile:
                prod_wells_input = get_prod_wells_input_fnn(
                    sim_output, resized_states_dict, perm)
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.CALLS)
                    .print_stats()
                )
        else:
            prod_wells_input = get_prod_wells_input_fnn(
                sim_output, resized_states_dict, perm)
            print("------- prod_wells_input: ", prod_wells_input.shape)

        inn_fcn.append(prod_wells_input)
        del (prod_wells_input)
        gc.collect()

        if PROFILE_CODE:
            print(f"---------- Profiling get_prod_wells_output_fnn --------------")
            with Profile() as profile:
                prod_wells_output = get_prod_wells_output_fnn(sim_output)
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.CALLS)
                    .print_stats()
                )
        else:
            prod_wells_output = get_prod_wells_output_fnn(sim_output)
            print("------- prod_wells_output: ", prod_wells_output.shape)

        out_fcn.append(prod_wells_output)
        del (prod_wells_output)
        gc.collect()

        Pr = resized_states_dict['PRESSURE']
        sw = resized_states_dict['SWAT']
        sg = resized_states_dict['SGAS']
        permx = resized_props_dict['PERMX']
        poro = resized_props_dict['PORO']
        print("Processed pressure: Len = ", len(Pr))
        print("Processed Swat: Len = ", len(sw))
        print("Processed Sgas: Len = ", len(sg))
        print("Processed time: Len = ", len(time_values_on_grid_cells))
        print("Processed flt mults: Len = ", len(flt))

        if PROFILE_CODE:
            print(f"---------- Profiling round_array_to_4dp --------------")
            with Profile() as profile:
                Pr = round_array_to_4dp(clip_and_convert_to_float32(Pr))
                sw = round_array_to_4dp(clip_and_convert_to_float32(sw))
                sg = round_array_to_4dp(clip_and_convert_to_float32(sg))
                permx = round_array_to_4dp(clip_and_convert_to_float32(permx))
                poro = round_array_to_4dp(clip_and_convert_to_float32(poro))
                time_values_on_grid_cells = round_array_to_4dp(
                    clip_and_convert_to_float32(time_values_on_grid_cells))
                flt = round_array_to_4dp(clip_and_convert_to_float32(flt))
                (
                    Stats(profile)
                    .strip_dirs()
                    .sort_stats(SortKey.CALLS)
                    .print_stats()
                )
        else:
            Pr = round_array_to_4dp(clip_and_convert_to_float32(Pr))
            sw = round_array_to_4dp(clip_and_convert_to_float32(sw))
            sg = round_array_to_4dp(clip_and_convert_to_float32(sg))
            permx = round_array_to_4dp(clip_and_convert_to_float32(permx))
            poro = round_array_to_4dp(clip_and_convert_to_float32(poro))
            time_values_on_grid_cells = round_array_to_4dp(
                clip_and_convert_to_float32(time_values_on_grid_cells))
            flt = round_array_to_4dp(clip_and_convert_to_float32(flt))

        permeability[i, 0, :, :, :] = permx
        porosity[i, 0, :, :, :] = poro

        active_cells_mask_array = grid_data['GRID DATA']['ACTNUM'][0]
        actnumm[i, 0, :, :, :] = np.reshape(
            active_cells_mask_array, (nx, ny, nz), "F")

        pressure.append(Pr)
        del (Pr)
        gc.collect()

        Sgas.append(sg)
        del (sg)
        gc.collect()

        Swater.append(sw)
        del (sw)
        gc.collect()

        Fault.append(flt)
        del (flt)
        gc.collect()
        del (permx)
        gc.collect()
        del (poro)
        gc.collect()

        ReportTimes.append(time_values_on_grid_cells)
        del (time_values_on_grid_cells)
        gc.collect()

    pressure = np.stack(pressure, axis=0)
    Sgas = np.stack(Sgas, axis=0)
    Swater = np.stack(Swater, axis=0)
    Fault = np.stack(Fault, axis=0)[:, None, :, :, :]
    ReportTimes = np.stack(ReportTimes, axis=0)
    inn_fcn = np.stack(inn_fcn, axis=0)
    out_fcn = np.stack(out_fcn, axis=0)
    print("------- Stacked out_fcn: ", out_fcn.shape)
    inn_fcn[np.isnan(inn_fcn)] = 0.0
    out_fcn[np.isnan(out_fcn)] = 0.0

    print("|-----------------------------------------------------------------|")
    print("|                 FINISHED DATA CURATION                          |")
    print("|-----------------------------------------------------------------|")

    Qw, Qg, Qo = get_injector_data(grid_data, sim_output)
    Q = Qw + Qg + Qo

    permeability[np.isnan(permeability)] = 0.0
    ReportTimes[np.isnan(ReportTimes)] = 0.0
    pressure[np.isnan(pressure)] = 0.0
    Qw[np.isnan(Qw)] = 0.0
    Qg[np.isnan(Qg)] = 0.0
    Q[np.isnan(Q)] = 0.0

    permeability[np.isinf(permeability)] = 0.0
    ReportTimes[np.isinf(ReportTimes)] = 0.0
    pressure[np.isinf(pressure)] = 0.0
    Qw[np.isinf(Qw)] = 0.0
    Qg[np.isinf(Qg)] = 0.0
    Q[np.isinf(Q)] = 0.0

    minK, maxK, permeabilityx = normalize(permeability)  # Permeability
    minT, maxT, Timex = normalize(ReportTimes)  # ReportTimes
    minP, maxP, pressurex = normalize(pressure)  # pressure
    minQw, maxQw, Qwx = normalize(Qw)  # Qw
    minQg, maxQg, Qgx = normalize(Qg)  # Qg
    minQ, maxQ, Qx = normalize(Q)  # Q

    permeabilityx[np.isnan(permeabilityx)] = 0.0
    Timex[np.isnan(Timex)] = 0.0
    pressurex[np.isnan(pressurex)] = 0.0

    Qwx[np.isnan(Qwx)] = 0.0
    Qgx[np.isnan(Qgx)] = 0.0
    Qx[np.isnan(Qx)] = 0.0

    permeabilityx[np.isinf(permeabilityx)] = 0.0
    Timex[np.isinf(Timex)] = 0.0
    pressurex[np.isinf(pressurex)] = 0.0

    Qwx[np.isinf(Qwx)] = 0.0
    Qgx[np.isinf(Qgx)] = 0.0
    Qx[np.isinf(Qx)] = 0.0

    porosity[np.isnan(porosity)] = 0.0
    Fault[np.isnan(Fault)] = 0.0
    Swater[np.isnan(Swater)] = 0.0
    Sgas[np.isnan(Sgas)] = 0.0
    actnumm[np.isnan(actnumm)] = 0.0

    porosity[np.isinf(porosity)] = 0.0
    Fault[np.isinf(Fault)] = 0.0
    Swater[np.isinf(Swater)] = 0.0
    Sgas[np.isinf(Sgas)] = 0.0
    actnumm[np.isinf(actnumm)] = 0.0

    ini_pressure, ini_sat = get_initial_pressure_saturation(
        maxP, nx, ny, nz, N_ens)

    X_data1 = {
        "permeability": permeabilityx,
        "porosity": porosity,
        "Pressure": pressurex,
        "Fault": Fault,
        "Water_saturation": Swater,
        "Time": Timex,
        "Gas_saturation": Sgas,
        "actnum": actnumm,
        "Qw": Qwx,
        "Qg": Qgx,
        "Q": Qx,
        "Pini": ini_pressure,
        "Sini": ini_sat
    }

    X_data1 = clean_dict_arrays(X_data1)

    del permeabilityx
    gc.collect()
    del permeability
    gc.collect()
    del porosity
    gc.collect()
    del pressurex
    gc.collect()
    del Fault
    gc.collect()
    del Swater
    gc.collect()
    del Timex
    gc.collect()
    del Sgas
    gc.collect()
    del actnumm
    gc.collect()
    del pressure
    gc.collect()
    del ReportTimes
    gc.collect()
    del Qwx
    gc.collect()
    del Qgx
    gc.collect()
    del Qx
    gc.collect()

    # Save compressed dictionary
    os.makedirs(dirname(simulation_train_data_file_path), exist_ok=True)
    print("---------------- Train file save path:  ",
          simulation_train_data_file_path)
    if PROFILE_CODE:
        print(f"---------- Profiling pickle dump data_train.pkl.gz --------------")
        with Profile() as profile:
            pickle_dump(simulation_train_data_file_path, X_data1)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        pickle_dump(simulation_train_data_file_path, X_data1)

    os.makedirs(dirname(simulation_test_data_file_path), exist_ok=True)

    if PROFILE_CODE:
        print(f"---------- Profiling pickle dump data_test.pkl.gz --------------")
        with Profile() as profile:
            pickle_dump(simulation_test_data_file_path, X_data1)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        pickle_dump(simulation_test_data_file_path, X_data1)

    min_inn_fcn, max_inn_fcn, inn_fcnx = normalize(inn_fcn)
    min_out_fcn, max_out_fcn, out_fcnx = normalize(out_fcn)

    inn_fcnx = clip_and_convert_to_float32(inn_fcnx)
    out_fcnx = clip_and_convert_to_float32(out_fcnx)

    print("------- inn_fcnx: ", inn_fcnx.shape)
    print("------- out_fcnx: ", out_fcnx.shape)

    X_data2 = {"X": inn_fcnx, "Y": out_fcnx}
    for key in X_data2.keys():
        X_data2[key][np.isnan(X_data2[key])] = 0.0  # Convert NaN to 0
        X_data2[key][np.isinf(X_data2[key])] = 0.0  # Convert infinity to 0
        # X_data2[key] = np.clip(X_data2[key], target_min, target_max)

    del inn_fcnx
    gc.collect()
    del inn_fcn
    gc.collect()
    del out_fcnx
    gc.collect()
    del out_fcn
    gc.collect()

    os.makedirs(dirname(simulation_train_wells_data_file_path), exist_ok=True)
    if PROFILE_CODE:
        print(f"---------- Profiling pickle dump data_train_peaceman.pkl.gz --------------")
        with Profile() as profile:
            pickle_dump(simulation_train_wells_data_file_path, X_data2)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        pickle_dump(simulation_train_wells_data_file_path, X_data2)

    os.makedirs(dirname(simulation_test_wells_data_file_path), exist_ok=True)
    if PROFILE_CODE:
        print(f"---------- Profiling pickle dump data_test_peaceman.pkl.gz --------------")
        with Profile() as profile:
            pickle_dump(simulation_test_wells_data_file_path, X_data2)
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        pickle_dump(simulation_test_wells_data_file_path, X_data2)

    os.makedirs(dirname(conversions_file_path), exist_ok=True)
    if PROFILE_CODE:
        print(f"---------- Profiling scipy_io.savemat --------------")
        with Profile() as profile:
            scipy_io.savemat(
                conversions_file_path,
                {
                    "minK": minK,
                    "maxK": maxK,
                    "minT": minT,
                    "maxT": maxT,
                    "minP": minP,
                    "maxP": maxP,
                    "minQW": minQw,
                    "maxQW": maxQw,
                    "minQg": minQg,
                    "maxQg": maxQg,
                    "minQ": minQ,
                    "maxQ": maxQ,
                    "min_inn_fcn": min_inn_fcn,
                    "max_inn_fcn": max_inn_fcn,
                    "min_out_fcn": min_out_fcn,
                    "max_out_fcn": max_out_fcn,
                    "steppi": len(time_slices_to_include),
                    "time_slices_to_include": time_slices_to_include,
                    "N_ens": N_ens,
                },
                do_compression=True,
            )
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        scipy_io.savemat(
            conversions_file_path,
            {
                "minK": minK,
                "maxK": maxK,
                "minT": minT,
                "maxT": maxT,
                "minP": minP,
                "maxP": maxP,
                "minQW": minQw,
                "maxQW": maxQw,
                "minQg": minQg,
                "maxQg": maxQg,
                "minQ": minQ,
                "maxQ": maxQ,
                "min_inn_fcn": min_inn_fcn,
                "max_inn_fcn": max_inn_fcn,
                "min_out_fcn": min_out_fcn,
                "max_out_fcn": max_out_fcn,
                "steppi": len(time_slices_to_include),
                "time_slices_to_include": time_slices_to_include,
                "N_ens": N_ens,
                "nx": nx,
                "ny": ny,
                "nz": nz,
            },
            do_compression=True,
        )

    print("|-----------------------------------------------------------------|")
    print("|                 DATA SAVED                                      |")
    print("|-----------------------------------------------------------------|")

    if DELETE_DIRECTORIES:
        print("|-----------------------------------------------------------------|")
        print("|                 REMOVING FOLDERS USED FOR THE RUN               |")
        print("|-----------------------------------------------------------------|")
        folder_prefix = ensemble_dir_root_name + realization_prefix_dirname
        for ens in range(N_ens):
            folder_ens = abspath(folder_prefix + str(ens))
            rmtree(folder_ens)


def test_fault_file_read(ensemble_dir_root_name) -> None:
    fault_file_path = base_dir + "faultensemble.dat"
    if PROFILE_CODE:
        print(f"---------- Profiling genfromtxt --------------")
        with Profile() as profile:
            fault_ensemble = np.genfromtxt(abspath(fault_file_path))
            (
                Stats(profile)
                .strip_dirs()
                .sort_stats(SortKey.CALLS)
                .print_stats()
            )
    else:
        fault_ensemble = np.genfromtxt(abspath(fault_file_path))


def test():
    curdir = "/workspace/project/src/standalone-scripts/rev-1/"
    os.chdir(curdir)
    generate_data(ensemble_dir_root_name,
                  realization_prefix_dirname, base_filename, Nens)
    os.chdir(curdir)


if __name__ == "__main__":
    # cProfile.run('test()')
    test()
