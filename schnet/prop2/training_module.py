import os
import schnetpack as spk
import schnetpack.transform as trn

import sys

import torch
import torchmetrics
import pytorch_lightning as pl
from schnetpack.data import AtomsDataModule, AtomsDataFormat


import numpy as np
np.bool = np.bool_
from ase import Atoms

import random
import pandas as pd

log_data = './log_data'
db_path = '../input_processing/upto_4500.db'  #  location of database file
data_set_size = 4500
epochs = 1000

if __name__ == "__main__":

    if not os.path.exists(log_data):
        os.makedirs(log_data)

    try:
        os.remove('./split.npz')
    except FileNotFoundError:
        pass
    #############################################
    ############ Data File preparations #########
    #############################################
    data_set = AtomsDataModule(
        db_path,
        format = AtomsDataFormat.ASE,
        batch_size = 500,
        num_train = 4000,
        num_val = 500,
        transforms = [
            trn.ASENeighborList(cutoff=5.),
            # pre processing of data
            trn.RemoveOffsets('prop2', remove_mean=True),
            trn.CastTo32()      # for faster training in GPU
            ],
        num_workers = 47,
        pin_memory = True,   # set false when not using GPU
        # load data to check its means and Std parameter
        load_properties = ['prop2'],
        property_units = {'prop2': 'Ha'}
    )

    data_set.prepare_data()
    data_set.setup()


    #######################################
    #########  Setting up the model  ######
    #######################################



    # 3 interaction layers, 5A cosine cutoff, 20 Gaussians, 50 atomwise features
    n_interactions = 3
    cutoff = 5.
    n_rbf = 20
    n_atom_basis= 50

    # calculates the pairwie distance between atoms
    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf = n_rbf, cutoff = cutoff)
    schnet = spk.representation.SchNet(
            n_atom_basis = n_atom_basis,
            n_interactions = n_interactions,
            radial_basis = radial_basis,
            cutoff_fn =spk.nn.CosineCutoff(cutoff)
    )
    pred_prop1 = spk.atomistic.Atomwise(n_in = n_atom_basis, output_key = 'prop2')

    nnpot = spk.model.NeuralNetworkPotential(
            representation = schnet,
            input_modules = [pairwise_distance],
            output_modules = [pred_prop1],
            postprocessors = [trn.CastTo64(), trn.AddOffsets('prop2', add_mean=True)]
    )

    #output model parameters
    output_prop1 = spk.task.ModelOutput(
            name = 'prop2',
            loss_fn = torch.nn.MSELoss(),
            loss_weight = 1.,
            metrics = { 'MAE': torchmetrics.MeanAbsoluteError()
            }
    )

    # Pass all infotmation to AtomisticTask
    task = spk.task.AtomisticTask(
            model = nnpot,
            outputs = [output_prop1],
            optimizer_cls = torch.optim.AdamW,
            optimizer_args={'lr':1e-3}
    )


    #############################################
    ################## Training the Model #######
    #############################################


    logger = pl.loggers.TensorBoardLogger(save_dir = log_data)
    callbacks = [
            spk.train.ModelCheckpoint(
                model_path = os.path.join(log_data, 'best_inference_model'),
                save_top_k = 1,
                monitor = 'val_loss'
            )
    ]

    trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            callbacks = callbacks,
            logger = logger,
            default_root_dir = log_data,
            max_epochs = epochs
    )

    trainer.fit(task, datamodule = data_set)

    ##################################################
    #################### Inference ###################
    ##################################################

    best_model = torch.load(os.path.join(log_data, 'best_inference_model'), map_location= 'cpu')

    ####################################################
    ############   Prediction on the own data set ######
    ####################################################
    try:
        test_data = np.load('../input_processing/test_data_file_500.npz',
                mmap_mode='r',  allow_pickle=True)
    except FileNotFoundError:
        sys.exit( "Given npz file not exist on the directory. Redo with valid file.")
    
    try:
        train_data = np.load('../input_processing/data_file_upto_4500.npz',
                mmap_mode='r',  allow_pickle=True)
    except FileNotFoundError:
        sys.exit( "Given npz file not exist on the directory. Redo with valid file.")

    ### alternative process to predict, # ASE automatically converts energy unit to eV
    calculator = spk.interfaces.SpkCalculator(
            model_file = os.path.join(log_data, 'best_inference_model'), # path to model
            neighbor_list = trn.ASENeighborList(cutoff=5.), #Neighbour List
            energy_key = 'prop2',  # name of energy property in model
            energy_unit = 'Ha', # unit of the property
            device = 'cuda',    # computation device
            )


    correlation_file = os.path.join(log_data, 'testing_AccVsPred_prop2.dat')
    try:
        os.remove(corelation_file)
    except NameError:
        pass

    corr = []

    for i in range(500):
        tmp = []
        symbols, positions  = test_data["Z"][i], test_data["R"][i]
        prop1_acc = test_data["prop2"][i] 
        atoms = Atoms(symbols = symbols, positions = positions)

        tmp.append(prop1_acc[0])
        
        atoms.set_calculator(calculator)
        pred1 = (atoms.get_total_energy()) * 0.0367493       # Transform back eV to Ha unit

        tmp.append(pred1)
        corr.append(tmp)
    
    corr = np.array(corr)
    np.savetxt(correlation_file, corr, delimiter =' ')


    correlation_file = os.path.join(log_data, 'training_AccVsPred_prop2.dat')
    try:
        os.remove(corelation_file)
    except NameError:
        pass

    corr = []

    for i in range(4500):
        tmp = []
        symbols, positions  = train_data["Z"][i], train_data["R"][i]
        prop1_acc = train_data["prop2"][i]
        atoms = Atoms(symbols = symbols, positions = positions)

        tmp.append(prop1_acc[0])

        atoms.set_calculator(calculator)
        pred1 = (atoms.get_total_energy()) * 0.0367493       # Transform back eV to Ha unit

        tmp.append(pred1)
        corr.append(tmp)

    corr = np.array(corr)
    np.savetxt(correlation_file, corr, delimiter =' ')

    print("Done")
