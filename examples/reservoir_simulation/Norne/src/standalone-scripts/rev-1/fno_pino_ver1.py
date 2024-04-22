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

# Directories and filenames declarations
root_dir = "/workspace/project/src/standalone-scripts/rev-1/"
base_dir = root_dir + "test1/"
data_dir = base_dir + "ensemble/nn_data/"
loss_file = base_dir + "losses.csv"

conversions_file_path = abspath(
    data_dir + "conversions.mat")

res_grid_static_data_file_path_hdf5 = abspath(
    data_dir + "simulations_res_grid_static_data.hdf5")

res_grid_train_file_path_hdf5 = abspath(
    data_dir + "simulations_res_grid_train.hdf5")
res_grid_test_file_path_hdf5 = abspath(
    data_dir + "simulations_res_grid_test.hdf5")

res_wells_train_file_path_hdf5 = abspath(
    data_dir + "simulations_wells_train.hdf5")
res_wells_test_file_path_hdf5 = abspath(
    data_dir + "simulations_wells_test.hdf5")


@modulus.sym.main(config_path="conf", config_name="config_PINO")
def setup_and_solve(cfg: ModulusConfig):
    """
    This is the main solve function 
    """
    mat = scipy_io.loadmat(conversions_file_path)
    steppi = mat["steppi"][0][0]
    N_ens = mat["N_ens"][0][0]

    reservoir_static_data = hdf5_load(res_grid_static_data_file_path_hdf5)
    reservoir_train_data = hdf5_load(res_grid_train_file_path_hdf5)
    reservoir_test_data = hdf5_load(res_grid_test_file_path_hdf5)
    prod_wells_train_data = hdf5_load(res_wells_train_file_path_hdf5)
    prod_wells_test_data = hdf5_load(res_wells_test_file_path_hdf5)

    printdict("invar_train",      reservoir_static_data)
    printdict("outvar_train1",  get_dict_by_key(
        reservoir_train_data, "PRESSURE"))
    printdict("outvar_train2",  get_dict_by_key(
        reservoir_train_data, "WATER SATURATION"))
    printdict("outvar_train3",  get_dict_by_key(
        reservoir_train_data, "GAS SATURATION"))
    printdict("invar_trainp",  get_dict_by_key(
        prod_wells_train_data, "AVG_PERM_PRESS_SAT_TIME"))
    printdict("outvar_trainp", get_dict_by_key(
        prod_wells_train_data, "PHASE PROD RATES"))

    # DictGridDataset expects a dictionary for the output variable.
    train_dataset_pressure = DictGridDataset(
        reservoir_static_data, get_dict_by_key(reservoir_train_data, "PRESSURE"))
    train_dataset_water = DictGridDataset(
        reservoir_static_data, get_dict_by_key(reservoir_train_data, "WATER SATURATION"))
    train_dataset_gas = DictGridDataset(
        reservoir_static_data, get_dict_by_key(reservoir_train_data, "GAS SATURATION"))
    train_dataset_wells = DictGridDataset(
        get_dict_by_key(prod_wells_train_data, "AVG_PERM_PRESS_SAT_TIME"),
        get_dict_by_key(prod_wells_train_data, "PHASE PROD RATES"))

    printdict("invar_test",      reservoir_static_data)
    printdict("outvar_test1",  get_dict_by_key(
        reservoir_test_data, "PRESSURE"))
    printdict("outvar_test2",  get_dict_by_key(
        reservoir_test_data, "WATER SATURATION"))
    printdict("outvar_test3",  get_dict_by_key(
        reservoir_test_data, "GAS SATURATION"))
    printdict("invar_testp",  get_dict_by_key(
        prod_wells_test_data, "AVG_PERM_PRESS_SAT_TIME"))
    printdict("outvar_testp", get_dict_by_key(
        prod_wells_test_data, "PHASE PROD RATES"))

    test_dataset_pressure = DictGridDataset(
        reservoir_static_data, get_dict_by_key(reservoir_train_data, "PRESSURE"))
    test_dataset_water = DictGridDataset(
        reservoir_static_data, get_dict_by_key(reservoir_train_data, "WATER SATURATION"))
    test_dataset_gas = DictGridDataset(
        reservoir_static_data, get_dict_by_key(reservoir_train_data, "GAS SATURATION"))
    test_dataset_wells = DictGridDataset(
        get_dict_by_key(prod_wells_test_data, "AVG_PERM_PRESS_SAT_TIME"),
        get_dict_by_key(prod_wells_test_data, "PHASE PROD RATES"))

    # input_keys = [
    #     Key("PERM-XDIR", size=1),
    #     Key("PORO", size=1),
    #     Key("INITIAL PRESSURE", size=1),
    #     Key("INITIAL WATER SAT", size=1),
    #     Key("FAULT TRANS MULT", size=1),
    # ]
    input_keys = get_modulus_input_keys(reservoir_static_data, None, None)

    # Pressure
    decoder1 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("PRESSURE", size=8)]
    )
    fno_pressure = FNOArch(
        input_keys,
        dimension=3,
        decoder_net=decoder1,
    )
    fno_pressure = fno_pressure.to(device)

    decoder2 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("WATER SATURATION", size=8)]
    )
    fno_water = FNOArch(
        input_keys,
        dimension=3,
        decoder_net=decoder2,
    )
    fno_water = fno_water.to(device)

    decoder3 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("GAS SATURATION", size=8)]
    )
    fno_gas = FNOArch(
        input_keys,
        dimension=3,
        decoder_net=decoder3,
    )
    fno_gas = fno_gas.to(device)

    decoder4 = ConvFullyConnectedArch(
        [Key("z", size=32)], [Key("PHASE PROD RATES", size=66)])
    fno_wells = FNOArch(
        [Key("AVG_PERM_PRESS_SAT_TIME", size=90)],
        fno_modes=12,
        dimension=1,
        padding=20,
        nr_fno_layers=5,
        decoder_net=decoder4,
    )
    fno_wells = fno_wells.to(device)

    nodes = (
        [fno_pressure.make_node("fno_forward_model_pressure")]
        + [fno_water.make_node("fno_forward_model_water")]
        + [fno_gas.make_node("fno_forward_model_gas")]
        + [fno_wells.make_node("fno_forward_model_peacemann")]
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
        dataset=train_dataset_wells,
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
        dataset=test_dataset_wells,
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
