from util_functions import *
from modulus.sym.models.fno import *
from modulus.sym.models.fully_connected import *
from modulus.sym.key import Key
from modulus.sym.hydra import ModulusConfig
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.domain.constraint import SupervisedGridConstraint
from modulus.sym.domain.validator import GridValidator, PointwiseValidator
from modulus.sym.dataset import DictGridDataset

import modulus
import cupy as cp
import numpy as np
import os
from os.path import abspath
import gzip
import pickle
import gc
import time
from datetime import timedelta
import matplotlib.pyplot as plt
import scipy.io as scipy_io

import csv

# torch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float32)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

PHYSICS_INFORMED_DOMAIN = False
PHYSICS_INFORMED_WELL = False
USE_PYTORCH_FRAMEWORK = False
HDF5_DUMP = True if not USE_PYTORCH_FRAMEWORK else False

# Directories and filenames declarations
root_dir = "/workspace/project/src/standalone-scripts/rev-1/"
base_dir = root_dir + "test1/"
data_dir = base_dir + "ensemble/PACKETS/"
loss_file = base_dir + "losses.csv"

simulation_train_data_file_path = abspath(data_dir + "data_train.pkl.gz")
simulation_test_data_file_path = abspath(data_dir + "data_test.pkl.gz")
simulation_train_wells_data_file_path = abspath(
    data_dir + "data_train_peaceman.pkl.gz")
simulation_test_wells_data_file_path = abspath(
    data_dir + "data_test_peaceman.pkl.gz")

train_file_path = abspath(data_dir + "simulations_train.pkl.gz")
train_file_path_hdf5 = abspath(data_dir + "simulations_train.hdf5")

test_file_path = abspath(data_dir + "simulations_test.pkl.gz")
test_file_path_hdf5 = abspath(data_dir + "simulations_test.hdf5")

peacemann_train_file_path = abspath(data_dir + "peacemann_train.mat")
peacemann_train_file_path_hdf5 = abspath(data_dir + "peacemann_train.hdf5")

peacemann_test_file_path = abspath(data_dir + "peacemann_test.mat")
peacemann_test_file_path_hdf5 = abspath(data_dir + "peacemann_test.hdf5")


class Labelledset:
    def __init__(self, data):
        self.data = {}
        for key in data.keys():
            self.data[key] = torch.from_numpy(data[key])

    def __getitem__(self, index):
        return {key: self.data[key][index].to(device, torch.float32)
                for key in self.data.keys()}

    def __len__(self):
        # Assuming the dictionary is not empty, get the first key.
        first_key = next(iter(self.data))
        # Return the length of the data associated with the first key.
        return len(self.data[first_key])


@modulus.sym.main(config_path="conf", config_name="config_PINO")
def setup_and_solve(cfg: ModulusConfig):

    SWOW = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOW), dtype=float)).to(
        device
    )

    SWOG = torch.tensor(np.array(np.vstack(cfg.custom.WELLSPECS.SWOG), dtype=float)).to(
        device
    )
    BO = float(cfg.custom.PROPS.BO)
    BW = float(cfg.custom.PROPS.BW)
    UW = float(cfg.custom.PROPS.UW)
    UO = float(cfg.custom.PROPS.UO)
    SWI = cp.float32(cfg.custom.PROPS.SWI)
    SWR = cp.float32(cfg.custom.PROPS.SWR)
    CFO = cp.float32(cfg.custom.PROPS.CFO)
    p_atm = cp.float32(float(cfg.custom.PROPS.PATM))
    p_bub = cp.float32(float(cfg.custom.PROPS.PB))

    params1 = {
        "BO": torch.tensor(BO),
        "UO": torch.tensor(UO),
        "BW": torch.tensor(BW),
        "UW": torch.tensor(UW),
    }

    skin = torch.tensor(0).to(device)
    rwell = torch.tensor(200).to(device)
    pwf_producer = torch.tensor(100).to(device)
    RE = torch.tensor(0.2 * 100).to(device)

    SWI = torch.from_numpy(np.array(SWI)).to(device)
    SWR = torch.from_numpy(np.array(SWR)).to(device)
    UW = torch.from_numpy(np.array(UW)).to(device)
    BW = torch.from_numpy(np.array(BW)).to(device)
    UO = torch.from_numpy(np.array(UO)).to(device)
    BO = torch.from_numpy(np.array(BO)).to(device)

    # SWOW = torch.from_numpy(SWOW).to(device)
    # SWOG = torch.from_numpy(SWOG).to(device)
    p_bub = torch.from_numpy(np.array(p_bub)).to(device)
    p_atm = torch.from_numpy(np.array(p_atm)).to(device)
    CFO = torch.from_numpy(np.array(CFO)).to(device)

    conv_file_path = abspath(data_dir + "conversions.mat")
    mat = scipy_io.loadmat(conv_file_path)
    minK = mat["minK"]
    maxK = mat["maxK"]
    minT = mat["minT"]
    maxT = mat["maxT"]
    minP = mat["minP"]
    maxP = mat["maxP"]
    minQw = mat["minQW"]
    maxQw = mat["maxQW"]
    minQg = mat["minQg"]
    maxQg = mat["maxQg"]
    minQ = mat["minQ"]
    maxQ = mat["maxQ"]
    min_inn_fcn = mat["min_inn_fcn"][0]
    max_inn_fcn = mat["max_inn_fcn"][0]
    min_out_fcn = mat["min_out_fcn"][0]
    max_out_fcn = mat["max_out_fcn"][0]
    steppi = mat["steppi"][0][0]
    time_slices_to_include = mat["time_slices_to_include"][0]
    N_ens = mat["N_ens"][0][0]
    nx = mat["nx"][0][0]
    ny = mat["ny"][0][0]
    nz = mat["nz"][0][0]

    # target_min = 0.01
    # target_max = 1

    # print("Values from conversion mat")
    # print("minK value is:", minK)
    # print("maxK value is:", maxK)
    # print("minT value is:", minT)
    # print("maxT value is:", maxT)
    # print("minP value is:", minP)
    # print("maxP value is:", maxP)
    # print("minQw value is:", minQw)
    # print("maxQw value is:", maxQw)
    # print("minQg value is:", minQg)
    # print("maxQg value is:", maxQg)
    # print("minQ value is:", minQ)
    # print("maxQ value is:", maxQ)
    # print("min_inn_fcn value is:", min_inn_fcn)
    # print("max_inn_fcn value is:", max_inn_fcn)
    # print("min_out_fcn value is:", min_out_fcn)
    # print("max_out_fcn value is:", max_out_fcn)
    # print("target_min value is:", target_min)
    # print("target_max value is:", target_max)
    # print("N_ens:", N_ens)
    # print("nx:", nx)
    # print("ny:", ny)
    # print("nz:", nz)
    # print("steppi:", steppi)
    # print("time_slices_to_include:", time_slices_to_include)

    # minKx = torch.from_numpy(minK).to(device)
    # maxKx = torch.from_numpy(maxK).to(device)
    # minTx = torch.from_numpy(minT).to(device)
    # maxTx = torch.from_numpy(maxT).to(device)
    # minPx = torch.from_numpy(minP).to(device)
    # maxPx = torch.from_numpy(maxP).to(device)
    # minQx = torch.from_numpy(minQ).to(device)
    # maxQx = torch.from_numpy(maxQ).to(device)
    # minQgx = torch.from_numpy(minQg).to(device)
    # maxQgx = torch.from_numpy(maxQg).to(device)
    # minQwx = torch.from_numpy(minQw).to(device)
    # maxQwx = torch.from_numpy(maxQw).to(device)
    # min_inn_fcnx = torch.from_numpy(min_inn_fcn).to(device)
    # max_inn_fcnx = torch.from_numpy(max_inn_fcn).to(device)
    # min_out_fcnx = torch.from_numpy(min_out_fcn).to(device)
    # max_out_fcnx = torch.from_numpy(max_out_fcn).to(device)

    del mat
    gc.collect()

    print("Load simulated labelled training data")
    with gzip.open(simulation_train_data_file_path, "rb") as f2:
        mat = pickle.load(f2)
    X_data1 = mat

    # for key, value in X_data1.items():
    #     print(f"For key '{key}':")
    #     print("\tContains inf:", np.isinf(value).any())
    #     print("\tContains -inf:", np.isinf(-value).any())
    #     print("\tContains NaN:", np.isnan(value).any())

    del mat
    gc.collect()

    for key in X_data1.keys():
        # Convert NaN and infinity values to 0
        X_data1[key][np.isnan(X_data1[key])] = 0.0
        X_data1[key][np.isinf(X_data1[key])] = 0.0
        X_data1[key] = clip_and_convert_to_float32(X_data1[key])

    cPerm = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    cPhi = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    cPini = np.zeros((N_ens, 1, nz, nx, ny),
                     dtype=np.float32)  # Initial pressure
    # Initial water saturation
    cSini = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)
    cfault = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Fault
    cQ = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQw = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQg = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cTime = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cactnum = np.zeros((1, 1, nz, nx, ny), dtype=np.float32)  # Fault

    cPress = np.zeros((N_ens, steppi, nz, nx, ny),
                      dtype=np.float32)  # Pressure
    cSat = np.zeros((N_ens, steppi, nz, nx, ny),
                    dtype=np.float32)  # Water saturation
    cSatg = np.zeros((N_ens, steppi, nz, nx, ny),
                     dtype=np.float32)  # gas saturation

    X_data1["Q"][X_data1["Q"] <= 0] = 0
    X_data1["Qw"][X_data1["Qw"] <= 0] = 0
    X_data1["Qg"][X_data1["Qg"] <= 0] = 0

    for i in range(nz):
        X_data1["Q"][0, :, :, :, i] = np.where(
            X_data1["Q"][0, :, :, :, i] < 0, 0, X_data1["Q"][0, :, :, :, i]
        )
        X_data1["Qw"][0, :, :, :, i] = np.where(
            X_data1["Qw"][0, :, :, :, i] < 0, 0, X_data1["Qw"][0, :, :, :, i]
        )
        X_data1["Qg"][0, :, :, :, i] = np.where(
            X_data1["Qg"][0, :, :, :, i] < 0, 0, X_data1["Qg"][0, :, :, :, i]
        )

        cQ[0, :, i, :, :] = X_data1["Q"][0, :, :, :, i]
        cQw[0, :, i, :, :] = X_data1["Qw"][0, :, :, :, i]
        cQg[0, :, i, :, :] = X_data1["Qg"][0, :, :, :, i]
        cactnum[0, 0, i, :, :] = X_data1["actnum"][0, 0, :, :, i]
        cTime[0, :, i, :, :] = X_data1["Time"][0, :, :, :, i]

    neededM = {
        "Q": torch.from_numpy(cQ).to(device, torch.float32),
        "Qw": torch.from_numpy(cQw).to(device, dtype=torch.float32),
        "Qg": torch.from_numpy(cQg).to(device, dtype=torch.float32),
        "actnum": torch.from_numpy(cactnum).to(device, dtype=torch.float32),
        "Time": torch.from_numpy(cTime).to(device, dtype=torch.float32),
    }

    for key in neededM:
        neededM[key] = replace_nans_and_infs(neededM[key])

    for kk in range(N_ens):

        # INPUTS
        for i in range(nz):
            cPerm[kk, 0, i, :, :] = clip_and_convert_to_float32(
                X_data1["permeability"][kk, 0, :, :, i]
            )
            cfault[kk, 0, i, :, :] = clip_and_convert_to_float32(
                X_data1["Fault"][kk, 0, :, :, i]
            )
            cPhi[kk, 0, i, :, :] = clip_and_convert_to_float32(
                X_data1["porosity"][kk, 0, :, :, i]
            )
            cPini[kk, 0, i, :, :] = (
                clip_and_convert_to_float32(
                    X_data1["Pini"][kk, 0, :, :, i]) / maxP
            )
            cSini[kk, 0, i, :, :] = clip_and_convert_to_float32(
                X_data1["Sini"][kk, 0, :, :, i]
            )

        # OUTPUTS
        for mum in range(steppi):
            for i in range(nz):
                cPress[kk, mum, i, :, :] = clip_and_convert_to_float32(
                    X_data1["Pressure"][kk, mum, :, :, i]
                )
                cSat[kk, mum, i, :, :] = clip_and_convert_to_float32(
                    X_data1["Water_saturation"][kk, mum, :, :, i]
                )
                cSatg[kk, mum, i, :, :] = clip_and_convert_to_float32(
                    X_data1["Gas_saturation"][kk, mum, :, :, i]
                )

    del X_data1
    gc.collect()
    data = {
        "perm": cPerm,
        "Phi": cPhi,
        "Pini": cPini,
        "Swini": cSini,
        "pressure": cPress,
        "water_sat": cSat,
        "gas_sat": cSatg,
        "fault": cfault,
    }

    if HDF5_DUMP == True:
        print("|-----------------------------------------------------------------|")
        print("|                 TRAINING DATA (FIELD) SAVING HDF5               |")
        print("|-----------------------------------------------------------------|")
        with gzip.open(train_file_path, "wb") as f4:
            pickle.dump(data, f4)

        preprocess_FNO_mat2(train_file_path, train_file_path_hdf5)
        # os.remove(train_file_path)
        del data
        gc.collect()

    del cPerm
    gc.collect()
    del cQ
    gc.collect()
    del cQw
    gc.collect()
    del cQg
    gc.collect()
    del cPhi
    gc.collect()
    del cTime
    gc.collect()
    del cPini
    gc.collect()
    del cSini
    gc.collect()
    del cPress
    gc.collect()
    del cSat
    gc.collect()
    del cSatg
    gc.collect()
    del cfault
    gc.collect()
    del cactnum
    gc.collect()

    print("Load simulated labelled test data from .gz file")
    with gzip.open(simulation_test_data_file_path, "rb") as f:
        mat = pickle.load(f)
    X_data1t = mat

    # for key, value in X_data1t.items():
    #     print(f"For key '{key}':")
    #     print("\tContains inf:", np.isinf(value).any())
    #     print("\tContains -inf:", np.isinf(-value).any())
    #     print("\tContains NaN:", np.isnan(value).any())

    del mat
    gc.collect()

    for key in X_data1t.keys():
        # Convert NaN and infinity values to 0
        X_data1t[key][np.isnan(X_data1t[key])] = 0
        X_data1t[key][np.isinf(X_data1t[key])] = 0
        # X_data1t[key] = np.clip(X_data1t[key], target_min, target_max)
        X_data1t[key] = clip_and_convert_to_float32(X_data1t[key])

    cPerm = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Permeability
    cPhi = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Porosity
    cPini = np.zeros((N_ens, 1, nz, nx, ny),
                     dtype=np.float32)  # Initial pressure
    cSini = np.zeros(
        (N_ens, 1, nz, nx, ny), dtype=np.float32
    )  # Initial water saturation
    cfault = np.zeros((N_ens, 1, nz, nx, ny), dtype=np.float32)  # Fault
    cQ = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQw = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cQg = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cTime = np.zeros((1, steppi, nz, nx, ny), dtype=np.float32)  # Fault
    cactnum = np.zeros((1, 1, nz, nx, ny), dtype=np.float32)  # Fault

    cPress = np.zeros((N_ens, steppi, nz, nx, ny),
                      dtype=np.float32)  # Pressure
    cSat = np.zeros((N_ens, steppi, nz, nx, ny),
                    dtype=np.float32)  # Water saturation
    cSatg = np.zeros((N_ens, steppi, nz, nx, ny),
                     dtype=np.float32)  # gas saturation

    for kk in range(N_ens):

        # INPUTS
        for i in range(nz):
            cPerm[kk, 0, i, :, :] = X_data1t["permeability"][kk, 0, :, :, i]
            cfault[kk, 0, i, :, :] = X_data1t["Fault"][kk, 0, :, :, i]
            cPhi[kk, 0, i, :, :] = X_data1t["porosity"][kk, 0, :, :, i]
            cPini[kk, 0, i, :, :] = X_data1t["Pini"][kk, 0, :, :, i] / maxP
            cSini[kk, 0, i, :, :] = X_data1t["Sini"][kk, 0, :, :, i]

        # OUTPUTS
        for mum in range(steppi):
            for i in range(nz):
                cPress[kk, mum, i, :, :] = X_data1t["Pressure"][kk, mum, :, :, i]
                cSat[kk, mum, i, :, :] = X_data1t["Water_saturation"][kk, mum, :, :, i]
                cSatg[kk, mum, i, :, :] = X_data1t["Gas_saturation"][kk, mum, :, :, i]
    del X_data1t
    gc.collect()

    data_test = {
        "perm": cPerm,
        "Phi": cPhi,
        "Pini": cPini,
        "Swini": cSini,
        "pressure": cPress,
        "water_sat": cSat,
        "gas_sat": cSatg,
        "fault": cfault,
    }

    if HDF5_DUMP == True:
        with gzip.open(test_file_path, "wb") as f4:
            pickle.dump(data_test, f4)

        preprocess_FNO_mat2(test_file_path, test_file_path_hdf5)
        # os.remove(test_file_path)
        del data_test
        gc.collect()

    del cPerm
    gc.collect()
    del cQ
    gc.collect()
    del cQw
    gc.collect()
    del cQg
    gc.collect()
    del cPhi
    gc.collect()
    del cTime
    gc.collect()
    del cPini
    gc.collect()
    del cSini
    gc.collect()
    del cPress
    gc.collect()
    del cSat
    gc.collect()
    del cSatg
    gc.collect()
    del cfault
    gc.collect()
    del cactnum
    gc.collect()

    print("|-----------------------------------------------------------------|")
    print("|                 PEACEMANN TRAIN DATA (WELL) SAVING              |")
    print("|-----------------------------------------------------------------|")

    print("Load simulated labelled training data for peacemann")
    with gzip.open(simulation_train_wells_data_file_path, "rb") as f:
        mat = pickle.load(f)
    X_data2 = mat

    del mat
    gc.collect()

    data2 = X_data2
    data2n = {key: value.transpose(0, 2, 1) for key, value in data2.items()}
    for key in data2n:
        data2n[key][data2n[key] <= 0] = 0

    del X_data2
    gc.collect()

    if HDF5_DUMP == True:
        scipy_io.savemat(peacemann_train_file_path,
                         data2n, do_compression=True)
        preprocess_FNO_mat(peacemann_train_file_path,
                           peacemann_train_file_path_hdf5)
        del data2
        gc.collect()
        del data2n
        gc.collect()

    print("|-----------------------------------------------------------------|")
    print("|                 PEACEMANN TEST DATA (WELL) SAVING               |")
    print("|-----------------------------------------------------------------|")
    print("Load simulated labelled test data for peacemann modelling")
    with gzip.open(simulation_test_wells_data_file_path, "rb") as f:
        mat = pickle.load(f)
    X_data2t = mat
    del mat
    gc.collect()

    data2_test = X_data2t
    data2n_test = {key: value.transpose(0, 2, 1)
                   for key, value in data2_test.items()}
    for key in data2n_test:
        data2n_test[key][data2n_test[key] <= 0] = 0

    if HDF5_DUMP == True:
        scipy_io.savemat(
            peacemann_test_file_path,
            data2n_test,
            do_compression=True,
        )
        preprocess_FNO_mat(peacemann_test_file_path,
                           peacemann_test_file_path_hdf5)
        del X_data2t
        gc.collect()
        del data2_test
        gc.collect()
        del data2n_test
        gc.collect()

    print("|-----------------------------------------------------------------|")
    print("|                 FINISHED DATA SETUP FOR NO                      |")
    print("|-----------------------------------------------------------------|")

    print("|-----------------------------------------------------------------|")
    print("|                 CREATING FNO NETWORKS                           |")
    print("|-----------------------------------------------------------------|")
    # Define FNO model
    # Pressure
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("pressure", size=steppi)]
    )
    fno_pressure = FNOArch(
        [
            Key("perm", size=1),
            Key("Phi", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
            Key("fault", size=1),
        ],
        dimension=3,
        decoder_net=decoder1,
    )
    # print("****************************************************************")
    # print("           fno_pressure                             ", fno_pressure)
    # print("****************************************************************")

    # fno_pressure = fno_pressure.to(device)

    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("water_sat", size=steppi)]
    )
    fno_water = FNOArch(
        [
            Key("perm", size=1),
            Key("Phi", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
            Key("fault", size=1),
        ],
        dimension=3,
        decoder_net=decoder2,
    )
    # fno_water = fno_water.to(device)

    decoder3 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("gas_sat", size=steppi)]
    )
    fno_gas = FNOArch(
        [
            Key("perm", size=1),
            Key("Phi", size=1),
            Key("Pini", size=1),
            Key("Swini", size=1),
            Key("fault", size=1),
        ],
        dimension=3,
        decoder_net=decoder3,
    )
    # fno_gas = fno_gas.to(device)

    decoder4 = ConvFullyConnectedArch([Key("z", size=32)], [Key("Y", size=66)])
    fno_peacemann = FNOArch(
        [Key("X", size=90)],
        fno_modes=13,
        dimension=1,
        padding=20,
        nr_fno_layers=5,
        decoder_net=decoder4,
    )
    # fno_peacemann = fno_peacemann.to(device)

    if USE_PYTORCH_FRAMEWORK == True:
        print("|-----------------------------------------------------------------|")
        print("|                 USING PYTORCH NATIVE SOLVER                     |")
        print("|-----------------------------------------------------------------|")

        learning_rate = cfg.optimizer.lr
        gamma = 0.5
        step_size = 100

        optimizer_pressure = torch.optim.AdamW(
            fno_pressure.parameters(),
            lr=learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
        scheduler_pressure = torch.optim.lr_scheduler.StepLR(
            optimizer_pressure, step_size=step_size, gamma=gamma
        )

        optimizer_water = torch.optim.AdamW(
            fno_water.parameters(),
            lr=learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
        scheduler_water = torch.optim.lr_scheduler.StepLR(
            optimizer_water, step_size=step_size, gamma=gamma
        )

        optimizer_gas = torch.optim.AdamW(
            fno_gas.parameters(), lr=learning_rate, weight_decay=cfg.optimizer.weight_decay
        )
        scheduler_gas = torch.optim.lr_scheduler.StepLR(
            optimizer_gas, step_size=step_size, gamma=gamma
        )

        optimizer_peacemann = torch.optim.AdamW(
            fno_peacemann.parameters(),
            lr=learning_rate,
            weight_decay=cfg.optimizer.weight_decay,
        )
        scheduler_peacemann = torch.optim.lr_scheduler.StepLR(
            optimizer_peacemann, step_size=step_size, gamma=gamma
        )

        ##############################################################################
        #         START THE TRAINING OF THE MODEL WITH UNLABELLED DATA - OPERATOR LEARNING
        ##############################################################################
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>OPERATOR LEARNING>>>>>>>>>>>>>>>>>>>>>>>>")
        train_total_loss_weighted_cpu_hist = []
        train_total_loss_cpu_hist = []
        test_total_loss_cpu_hist = []
        train_total_loss_weighted_cpu_updated = 0
        train_total_loss_cpu_updated = 0
        test_total_loss_cpu_updated = 0
        epochs = cfg.custom.epochs  # 'number of epochs to train'

        # myloss = LpLoss(size_average=True)
        dataset = Labelledset(data)
        labelled_loader = DataLoader(dataset, batch_size=cfg.batch_size.grid)

        imtest_use = 1
        index = np.random.choice(N_ens, imtest_use, replace=False)
        inn_test = {
            "perm": torch.from_numpy(data_test["perm"][index, :, :, :, :]).to(
                device, torch.float32
            ),
            "Phi": torch.from_numpy(data_test["Phi"][index, :, :, :, :]).to(
                device, dtype=torch.float32
            ),
            "fault": torch.from_numpy(data_test["fault"][index, :, :, :, :]).to(
                device, dtype=torch.float32
            ),
            "Pini": torch.from_numpy(data_test["Pini"][index, :, :, :, :]).to(
                device, dtype=torch.float32
            ),
            "Swini": torch.from_numpy(data_test["Swini"][index, :, :, :, :]).to(
                device, dtype=torch.float32
            ),
        }

        out_test = {
            "pressure": torch.from_numpy(data_test["pressure"][index, :, :, :, :]).to(
                device, torch.float32
            ),
            "water_sat": torch.from_numpy(data_test["water_sat"][index, :, :, :, :]).to(
                device, dtype=torch.float32
            ),
            "gas_sat": torch.from_numpy(data_test["gas_sat"][index, :, :, :, :]).to(
                device, dtype=torch.float32
            ),
        }

        inn_test_p = {
            "X": torch.from_numpy(data2n_test["X"][index, :, :]).to(device, torch.float32)
        }
        out_test_p = {
            "Y": torch.from_numpy(data2n_test["Y"][index, :, :]).to(device, torch.float32)
        }

        inn_train_p = {"X": torch.from_numpy(
            data2n["X"]).to(device, torch.float32)}
        out_train_p = {"Y": torch.from_numpy(
            data2n["Y"]).to(device, torch.float32)}

        print(" Training with " + str(N_ens) + " labelled members ")
        start_epoch = 1
        start_time = time.time()
        imtest_use = 1

        with open(loss_file, 'a') as file:
            # Write a header (optional, remove if appending to existing data)
            # file.write("train_total_loss, \
            #            test_total_loss, \
            #            train_pressure_loss, \
            #            train_watersat_loss, \
            #            train_gassat_loss, \
            #            test_wells_loss, \
            #            test_pressure_loss, \
            #            test_watersat_loss, \
            #            test_gassat_loss, \
            #            test_wells_loss, \
            #            \n")
            writer = csv.writer(file, delimiter=",")
            column_names = ["Train loss total (wgt)",
                            "Train loss total",
                            "Test loss total",
                            "Pressure train loss",
                            "Water sat train loss",
                            "Gas sat train loss",
                            "Wells train loss",
                            "Pressure test loss",
                            "Water sat test loss",
                            "Gas sat test loss",
                            "Wells test loss"]
            writer = csv.DictWriter(file, fieldnames=column_names)
            writer.writeheader()

            for epoch in range(start_epoch, epochs + 1):
                fno_pressure.train()
                fno_water.train()
                fno_gas.train()
                fno_peacemann.train()
                loss_train = 0.0
                print("Epoch " + str(epoch) + " | " + str(epochs))
                print("****************************************************************")
                for inputaa in labelled_loader:

                    inputin = {
                        "perm": inputaa["perm"],
                        "Phi": inputaa["Phi"],
                        "fault": inputaa["fault"],
                        "Pini": inputaa["Pini"],
                        "Swini": inputaa["Swini"],
                    }
                    target = {
                        "pressure": inputaa["pressure"],
                        "water_sat": inputaa["water_sat"],
                        "gas_sat": inputaa["gas_sat"],
                    }

                    optimizer_pressure.zero_grad()
                    optimizer_water.zero_grad()
                    optimizer_gas.zero_grad()
                    optimizer_peacemann.zero_grad()

                    outputa_p = fno_pressure(inputin)["pressure"]
                    outputa_s = fno_water(inputin)["water_sat"]
                    outputa_g = fno_gas(inputin)["gas_sat"]
                    outputa_pe = fno_peacemann(inn_train_p)["Y"]

                    # with torch.no_grad():
                    # TRAIN
                    dtrue_p = target["pressure"]
                    dtrue_s = target["water_sat"]
                    dtrue_g = target["gas_sat"]
                    dtrue_pe = out_train_p["Y"]

                    train_pressure_loss = MeanAbsoluteError(
                        (outputa_p).reshape(cfg.batch_size.labelled, -1),
                        (dtrue_p).reshape(cfg.batch_size.labelled, -1),
                    )

                    train_watersat_loss = MeanAbsoluteError(
                        (outputa_s).reshape(cfg.batch_size.labelled, -1),
                        (dtrue_s).reshape(cfg.batch_size.labelled, -1),
                    )

                    train_gassat_loss = MeanAbsoluteError(
                        (outputa_g).reshape(cfg.batch_size.labelled, -1),
                        (dtrue_g).reshape(cfg.batch_size.labelled, -1),
                    )

                    train_wells_loss = MeanAbsoluteError(
                        (outputa_pe).reshape(
                            N_ens, -1), (dtrue_pe).reshape(N_ens, -1)
                    )

                    train_total_loss = (
                        (train_pressure_loss * cfg.loss.weights.pressure)
                        + (train_watersat_loss * cfg.loss.weights.water_sat)
                        + (train_gassat_loss * cfg.loss.weights.gas_sat)
                        + train_wells_loss * cfg.loss.weights.Y
                    )

                    # print("****************************************************************")
                    # print("Loss:p ", train_pressure_loss)
                    # print("Loss:Sw ", train_watersat_loss)
                    # print("Loss:Sg ", train_gassat_loss)
                    # print("Loss:Wells ", train_wells_loss)
                    # print("Loss:Combined ", train_total_loss)
                    # print("****************************************************************")

                    # train_total_loss = torch.log10(train_total_loss)

                    # TEST

                    # input_varr = inputin
                    # input_varr["pressure"] = outputa_p
                    # input_varr["water_sat"] = outputa_s
                    # input_varr["gas_sat"] = outputa_g
#
                    # input_varr1 = inputin
                    # input_varr1["pressure"] = dtrue_p
                    # input_varr1["water_sat"] = dtrue_s
                    # input_varr1["gas_sat"] = dtrue_g

                    # print(outputa_p.shape)
                    # if PHYSICS_INFORMED_DOMAIN == True:
                    #     f_loss2, f_water2, f_gass2 = Black_oil(
                    #         neededM,
                    #         input_varr,
                    #         SWI,
                    #         SWR,
                    #         UW,
                    #         BW,
                    #         UO,
                    #         BO,
                    #         nx,
                    #         ny,
                    #         nz,
                    #         SWOW,
                    #         SWOG,
                    #         target_min,
                    #         target_max,
                    #         minKx,
                    #         maxKx,
                    #         minPx,
                    #         maxPx,
                    #         p_bub,
                    #         p_atm,
                    #         CFO,
                    #         maxQx,
                    #         maxQwx,
                    #         maxTx,
                    #         maxQgx,
                    #         Relperm,
                    #         params,
                    #         pde_method,
                    #         device,
                    #     )
        #
                    #     f_loss21, f_water21, f_gass21 = Black_oil(
                    #         neededM,
                    #         input_varr1,
                    #         SWI,
                    #         SWR,
                    #         UW,
                    #         BW,
                    #         UO,
                    #         BO,
                    #         nx,
                    #         ny,
                    #         nz,
                    #         SWOW,
                    #         SWOG,
                    #         target_min,
                    #         target_max,
                    #         minKx,
                    #         maxKx,
                    #         minPx,
                    #         maxPx,
                    #         p_bub,
                    #         p_atm,
                    #         CFO,
                    #         maxQx,
                    #         maxQwx,
                    #         maxTx,
                    #         maxQgx,
                    #         Relperm,
                    #         params,
                    #         pde_method,
                    #         device,
                    #     )
        #
                    #     loss_pde2 = f_loss2 + f_water2 + f_gass2 + f_loss21 + f_water21 + f_gass21
        #
                    # if PHYSICS_INFORMED_WELL == True:
                    #     loss_pdea = Black_oil_peacemann(
                    #         UO,
                    #         BO,
                    #         UW,
                    #         BW,
                    #         DZ,
                    #         RE,
                    #         inn_train_p["X"],
                    #         out_train_p["Y"],
                    #         device,
                    #         max_inn_fcnx,
                    #         max_out_fcnx,
                    #         params,
                    #         p_bub,
                    #         p_atm,
                    #         CFO,
                    #         device,
                    #     )
                    #     loss_pdeb = Black_oil_peacemann(
                    #         UO,
                    #         BO,
                    #         UW,
                    #         BW,
                    #         DZ,
                    #         RE,
                    #         inn_train_p["X"],
                    #         outputa_pe,
                    #         device,
                    #         max_inn_fcnx,
                    #         max_out_fcnx,
                    #         params,
                    #         p_bub,
                    #         p_atm,
                    #         CFO,
                    #         device,
                    #     )
        #
                    #     loss_pde = loss_pdea + loss_pdeb
        #
                    # loss_pe = Black_oil_peacemann(params,(inn_train_p)["X"],dtrue_pe,device,max_inn_fcn,max_out_fcn,params)

                    # train_total_loss_weighted = train_total_loss * cfg.custom.data_weighting * 1e5
                    # train_total_loss_weighted = normalize_tensors_adjusted(train_total_loss) * cfg.custom.data_weighting
                    train_total_loss_weighted = train_total_loss * cfg.custom.data_weighting

                    # if PHYSICS_INFORMED_DOMAIN == True:
                    #     train_total_loss_weighted += loss_pde2 * 1e-7
                    # if PHYSICS_INFORMED_WELL == True:
                    #     train_total_loss_weighted += loss_pde * 1e-7

                    model_pressure = fno_pressure
                    model_saturation = fno_water
                    model_gas = fno_gas
                    model_peacemann = fno_peacemann
                    train_total_loss_weighted.backward()

                    optimizer_pressure.step()
                    optimizer_water.step()
                    optimizer_gas.step()
                    optimizer_peacemann.step()

                    loss_train += train_total_loss_weighted.item()

                    with torch.no_grad():
                        outputa_ptest = fno_pressure(inn_test)["pressure"]
                        outputa_stest = fno_water(inn_test)["water_sat"]
                        outputa_gtest = fno_gas(inn_test)["gas_sat"]
                        outputa_petest = fno_peacemann(inn_test_p)["Y"]

                        dtest_p = out_test["pressure"]
                        dtest_s = out_test["water_sat"]
                        dtest_g = out_test["gas_sat"]
                        dtest_pe = out_test_p["Y"]

                        test_pressure_loss = MeanAbsoluteError(
                            (outputa_ptest).reshape(imtest_use, -1),
                            (dtest_p).reshape(imtest_use, -1),
                        )

                        test_watersat_loss = MeanAbsoluteError(
                            (outputa_stest).reshape(imtest_use, -1),
                            (dtest_s).reshape(imtest_use, -1),
                        )

                        test_gassat_loss = MeanAbsoluteError(
                            (outputa_gtest).reshape(imtest_use, -1),
                            (dtest_g).reshape(imtest_use, -1),
                        )

                        test_wells_loss = MeanAbsoluteError(
                            (outputa_petest).reshape(imtest_use, -1),
                            (dtest_pe).reshape(imtest_use, -1),
                        )

                        test_total_loss = test_pressure_loss + test_watersat_loss \
                            + test_gassat_loss + test_wells_loss

                scheduler_pressure.step()
                scheduler_water.step()
                scheduler_gas.step()
                scheduler_peacemann.step()

                train_total_loss_weighted_cpu = train_total_loss_weighted.detach().cpu().numpy()
                train_total_loss_weighted_cpu_hist.append(
                    train_total_loss_weighted_cpu)

                train_total_loss_cpu = train_total_loss.detach().cpu().numpy()
                train_total_loss_cpu_hist.append(train_total_loss_cpu)

                test_total_loss_cpu = test_total_loss.detach().cpu().numpy()
                test_total_loss_cpu_hist.append(test_total_loss_cpu)

                # Format the data as a comma-separated string
                # line = f"""{train_total_loss_weighted_cpu},
                # {train_total_loss_cpu},
                # {test_total_loss_cpu},
                # {train_pressure_loss.detach().cpu().numpy()},
                # {train_watersat_loss.detach().cpu().numpy()},
                # {train_gassat_loss.detach().cpu().numpy()},
                # {train_wells_loss.detach().cpu().numpy()},
                # {test_pressure_loss.detach().cpu().numpy()},
                # {test_watersat_loss.detach().cpu().numpy()},
                # {test_gassat_loss.detach().cpu().numpy()},
                # {test_wells_loss.detach().cpu().numpy()},
                # \n"""
                # file.write(line)

                writer.writerow({
                    "Train loss total (wgt)": train_total_loss_weighted_cpu,
                    "Train loss total": train_total_loss_cpu,
                    "Test loss total": test_total_loss_cpu,
                    "Pressure train loss": train_pressure_loss.detach().cpu().numpy(),
                    "Water sat train loss": train_watersat_loss.detach().cpu().numpy(),
                    "Gas sat train loss": train_gassat_loss.detach().cpu().numpy(),
                    "Wells train loss": train_wells_loss.detach().cpu().numpy(),
                    "Pressure test loss": test_pressure_loss.detach().cpu().numpy(),
                    "Water sat test loss": test_watersat_loss.detach().cpu().numpy(),
                    "Gas sat test loss": test_gassat_loss.detach().cpu().numpy(),
                    "Wells test loss": test_wells_loss.detach().cpu().numpy()
                })

                print("TRAINING")
                if train_total_loss_weighted_cpu_updated < train_total_loss_weighted_cpu:
                    print(
                        "   FORWARD PROBLEM COMMENT (Overall loss)) : Loss increased by "
                        + str(abs(train_total_loss_weighted_cpu_updated -
                              train_total_loss_weighted_cpu))
                    )

                elif train_total_loss_weighted_cpu_updated > train_total_loss_weighted_cpu:
                    print(
                        "   FORWARD PROBLEM COMMENT (Overall loss)) : Loss decreased by "
                        + str(abs(train_total_loss_weighted_cpu_updated -
                              train_total_loss_weighted_cpu))
                    )

                else:
                    print(
                        "   FORWARD PROBLEM COMMENT (Overall loss) : No change in Loss ")

                print("   training loss = " +
                      str(train_total_loss_weighted_cpu))
                print("   data loss = " +
                      str(train_total_loss.detach().cpu().numpy()))

                # print("   pde loss dynamic = " + str(loss_pde2.detach().cpu().numpy()))
                # print("   pde loss peacemann = " + str(loss_pde.detach().cpu().numpy()))
                # print("    ******************************   ")
                # print(
                #     "   Labelled pressure equation loss = "
                #     + str(f_loss2.detach().cpu().numpy())
                # )
                # print(
                #     "   Labelled saturation equation loss = "
                #     + str(f_water2.detach().cpu().numpy())
                # )
                # print("   Labelled gas equation loss = " +
                #       str(f_gass2.detach().cpu().numpy()))
                # print("    ******************************   ")
        #
                # print(
                #     "   unlabelled pressure equation loss = "
                #     + str(f_loss21.detach().cpu().numpy())
                # )
                # print(
                #     "   unlabelled saturation equation loss = "
                #     + str(f_water21.detach().cpu().numpy())
                # )
                # print(
                #     "   unlabelled gas equation loss = " +
                #     str(f_gass21.detach().cpu().numpy())
                # )
                # print("    ******************************   ")

                if train_total_loss_cpu_updated < train_total_loss_cpu:
                    print("   TRAINING COMMENT : Loss increased by " +
                          str(abs(train_total_loss_cpu_updated - train_total_loss_cpu)))
                elif train_total_loss_cpu_updated > train_total_loss_cpu:
                    print("   TRAINING COMMENT : Loss decreased by " +
                          str(abs(train_total_loss_cpu_updated - train_total_loss_cpu)))
                else:
                    print("   TRAINING COMMENT : No change in Loss ")

                print("   Train loss = " + str(train_total_loss_cpu))
                print("   Train: Pressure loss = " +
                      str(train_pressure_loss.detach().cpu().numpy()))
                print("   Train: saturation loss = " +
                      str(train_watersat_loss.detach().cpu().numpy()))
                print("   Train: gas loss = " +
                      str(train_gassat_loss.detach().cpu().numpy()))
                print("   Train: peacemann loss = " +
                      str(train_wells_loss.detach().cpu().numpy()))

                print("    ******************************   ")

                if test_total_loss_cpu_updated < test_total_loss_cpu:
                    print("   TEST COMMENT : Loss increased by " +
                          str(abs(test_total_loss_cpu_updated - test_total_loss_cpu)))
                elif test_total_loss_cpu_updated > test_total_loss_cpu:
                    print("   Test COMMENT : Loss decreased by " +
                          str(abs(test_total_loss_cpu_updated - test_total_loss_cpu)))
                else:
                    print("   TEST COMMENT : No change in Loss ")

                print("   Test loss = " + str(test_total_loss_cpu))
                print("   Test: Pressure loss = " +
                      str(test_pressure_loss.detach().cpu().numpy()))
                print("   Test: saturation loss = " +
                      str(test_watersat_loss.detach().cpu().numpy()))
                print("   Test: gas loss = " +
                      str(test_gassat_loss.detach().cpu().numpy()))
                print("   Test: peacemann loss = " +
                      str(test_wells_loss.detach().cpu().numpy()))

                print("    ******************************   ")

                train_total_loss_weighted_cpu_updated = train_total_loss_weighted_cpu
                train_total_loss_cpu_updated = train_total_loss_cpu
                test_total_loss_cpu_updated = test_total_loss_cpu

                if epoch == 1:
                    best_cost = train_total_loss_weighted_cpu_updated
                else:
                    pass
                if best_cost > train_total_loss_weighted_cpu:
                    print("    ******************************   ")
                    print("   Forward models saved")
                    print("   Current best cost = " + str(best_cost))
                    print("   Current epoch cost = " +
                          str(train_total_loss_weighted_cpu))
                    torch.save(model_pressure.state_dict(),
                               base_dir + "pressure_model.pth")
                    torch.save(model_saturation.state_dict(),
                               base_dir + "water_model.pth")
                    torch.save(model_gas.state_dict(),
                               base_dir + "gas_model.pth")
                    torch.save(model_peacemann.state_dict(),
                               base_dir + "peacemann_model.pth")
                    best_cost = train_total_loss_weighted_cpu
                else:
                    print("    ******************************   ")
                    print("   Forward models NOT saved")
                    print("   Current best cost = " + str(best_cost))
                    print("   Current epoch cost = " +
                          str(train_total_loss_weighted_cpu))

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print(">>>>>>>>>>>>>>>>>Finished training >>>>>>>>>>>>>>>>> ")
        elapsed_time_secs = time.time() - start_time
        msg = "PDE learning Execution took: %s secs (Wall clock time)" % timedelta(
            seconds=round(elapsed_time_secs)
        )
        print(msg)

        print("")
        plt.figure(figsize=(10, 10))

        plt.subplot(2, 2, 1)
        plt.semilogy(range(len(train_total_loss_weighted_cpu_hist)),
                     train_total_loss_weighted_cpu_hist, "k-")
        plt.title("Forward problem -Overall loss", fontsize=13)
        plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
        plt.xlabel("$n_{epoch}$", fontsize=13)

        plt.subplot(2, 2, 2)
        plt.semilogy(range(len(train_total_loss_cpu_hist)),
                     train_total_loss_cpu_hist, "k-")
        plt.title("Training loss", fontsize=13)
        plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
        plt.xlabel("$n_{epoch}$", fontsize=13)

        plt.subplot(2, 2, 3)
        plt.semilogy(range(len(test_total_loss_cpu_hist)),
                     test_total_loss_cpu_hist, "k-")
        plt.title("Test loss", fontsize=13)
        plt.ylabel("$\\phi_{n_{epoch}}$", fontsize=13)
        plt.xlabel("$n_{epoch}$", fontsize=13)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.suptitle("MODEL LEARNING", fontsize=16)
        plt.savefig(base_dir + "cost_PyTorch.png")
        plt.close()
        plt.clf()

    else:
        # load training/ test data for Numerical simulation model
        input_keys = [
            Key("perm"),
            Key("Phi"),
            Key("Pini"),
            Key("Swini"),
            Key("fault"),
        ]

        output_keys_pressure = [
            Key("pressure"),
        ]

        output_keys_water = [
            Key("water_sat"),
        ]

        output_keys_gas = [
            Key("gas_sat"),
        ]

        invar_train, outvar_train1, outvar_train2, outvar_train3 = load_FNO_dataset2(
            train_file_path_hdf5,
            [k.name for k in input_keys],
            [k.name for k in output_keys_pressure],
            [k.name for k in output_keys_water],
            [k.name for k in output_keys_gas],
            # n_examples=N_ens,
        )
        printdict("invar_train",      invar_train)
        printdict("outvar_train1",  outvar_train1)
        printdict("outvar_train2",  outvar_train2)
        printdict("outvar_train3",  outvar_train3)

        invar_test, outvar_test1, outvar_test2, outvar_test3 = load_FNO_dataset2(
            test_file_path_hdf5,
            [k.name for k in input_keys],
            [k.name for k in output_keys_pressure],
            [k.name for k in output_keys_water],
            [k.name for k in output_keys_gas],
            # n_examples=cfg.custom.ntest,
        )

        printdict("invar_test",     invar_test)
        printdict("outvar_test1", outvar_test1)
        printdict("outvar_test2", outvar_test2)
        printdict("outvar_test3", outvar_test3)

        # load training/ test data for peaceman model
        input_keysp = [Key("X")]
        output_keysp = [Key("Y")]

        # parse data
        invar_trainp, outvar_trainp = load_FNO_dataset2a(
            peacemann_train_file_path_hdf5,
            [k.name for k in input_keysp],
            [k.name for k in output_keysp],
            # n_examples=N_ens,
        )
        printdict("invar_trainp",  invar_trainp)
        printdict("outvar_trainp", outvar_trainp)

        invar_testp, outvar_testp = load_FNO_dataset2a(
            peacemann_test_file_path_hdf5,
            [k.name for k in input_keysp],
            [k.name for k in output_keysp],
            # n_examples=cfg.custom.ntest,
        )
        printdict("invar_testp",  invar_testp)
        printdict("outvar_testp", outvar_testp)

        # outvar_trainp["peacemanned"] = np.zeros_like(outvar_trainp["Y"])

        train_dataset_pressure = DictGridDataset(invar_train, outvar_train1)
        print("train_dataset_pressure: ", train_dataset_pressure.type,
              train_dataset_pressure.shape)
        train_dataset_water = DictGridDataset(invar_train, outvar_train2)
        train_dataset_gas = DictGridDataset(invar_train, outvar_train3)
        train_dataset_p = DictGridDataset(invar_trainp, outvar_trainp)

        test_dataset_pressure = DictGridDataset(invar_test, outvar_test1)
        test_dataset_water = DictGridDataset(invar_test, outvar_test2)
        test_dataset_gas = DictGridDataset(invar_test, outvar_test3)
        test_dataset_p = DictGridDataset(invar_testp, outvar_testp)

        nodes = (
            [fno_pressure.make_node("fno_forward_model_pressure")]
            + [fno_peacemann.make_node("fno_forward_model_peacemann")]
            + [fno_water.make_node("fno_forward_model_water")]
            + [fno_gas.make_node("fno_forward_model_gas")]
            # + [peacemannp]
        )  # + [darcyy]

        # make domain
        domain = Domain()

        # add constraints to domain
        supervised_dynamic_pressure = SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset_pressure,
            batch_size=1,
        )
        domain.add_constraint(supervised_dynamic_pressure,
                              "supervised_dynamic_pressure")

        supervised_dynamic_water = SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset_water,
            batch_size=1,
        )
        domain.add_constraint(supervised_dynamic_water,
                              "supervised_dynamic_water")

        supervised_dynamic_gas = SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset_gas,
            batch_size=1,
        )
        domain.add_constraint(supervised_dynamic_gas, "supervised_dynamic_gas")

        supervised_peacemann = SupervisedGridConstraint(
            nodes=nodes,
            dataset=train_dataset_p,
            batch_size=1,
        )
        domain.add_constraint(supervised_peacemann, "supervised_peacemann")

        # [constraint]
        # add validator

        test_dynamic_pressure = GridValidator(
            nodes,
            dataset=test_dataset_pressure,
            batch_size=cfg.batch_size.test,
            requires_grad=False,
        )
        domain.add_validator(test_dynamic_pressure, "test_dynamic_pressure")

        test_dynamic_water = GridValidator(
            nodes,
            dataset=test_dataset_water,
            batch_size=cfg.batch_size.test,
            requires_grad=False,
        )
        domain.add_validator(test_dynamic_water, "test_dynamic_water")

        test_dynamic_gas = GridValidator(
            nodes,
            dataset=test_dataset_gas,
            batch_size=cfg.batch_size.test,
            requires_grad=False,
        )
        domain.add_validator(test_dynamic_gas, "test_dynamic_gas")

        test_peacemann = GridValidator(
            nodes,
            dataset=test_dataset_p,
            batch_size=cfg.batch_size.test,
            requires_grad=False,
        )
        domain.add_validator(test_peacemann, "test_peaceman")

        # make solver
        slv = Solver(cfg, domain)

        # start solver
        slv.solve()


def test():
    Nens = 2
    print(root_dir)
    os.chdir(root_dir)
    # data_dir = './test/ensemble/PACKETS/'
    # test_fault_file_read(base_dir)
    print(abspath(data_dir))
    setup_and_solve()
    os.chdir(root_dir)


if __name__ == "__main__":
    # cProfile.run('test()')
    test()
