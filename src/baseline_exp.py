# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:52:26 2023

@author: Ying Fu
"""
import os
import random
import gc
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm
from matplotlib import pyplot as plt

import config
from dataloader.sequence_dataloader import SequenceDataset
from models.lstm import LSTM
from src.data.JetEngine import prepare_data
from train_evaluate import train_step, val_step, predict_step
from util import EarlyStopper, rmse_score, mae_score, mape_score

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def set_seed(seed):
    """
    Ensure Reproducibility. Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms.
    Furthermore, results may not be reproducible between CPU and GPU even using same seeds.
    Steps to limit the number of sources of nondeterministic behaviors.
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    torch.manual_seed(seed)  # PyTorch random number generator.
    np.random.seed(seed)  # random number on Numpy
    random.seed(seed)  # random seed
    torch.backends.cudnn.deterministic = True  # using deterministic algorithms
    # torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(train_loader, val_loader,
        train_step, val_step,
        model,
        n_epochs, device, loss_fn_regression, optimizer):
    """
    Executes the training and validation process for a given model.
    
    Args:
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        train_step (function): Function to execute a training step.
        val_step (function): Function to execute a validation step.
        model (torch.nn.Module): The model to train.
        n_epochs (int): Number of epochs to train the model.
        device (str): Device to run the training on ('cuda' or 'cpu').
        loss_fn_regression (function): Loss function for the regression task.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
    """
    avg_train_loss_lst = []
    avg_val_loss_lst = []

    early_stopper = EarlyStopper(patience=5, min_delta=0.001)
    for epoch in tqdm(range(n_epochs), desc="Training Epochs"):
        avg_train_loss = train_step(train_loader, device, model,
                                    loss_fn_regression, optimizer)
        avg_val_loss = val_step(val_loader, device, model, loss_fn_regression)
        avg_train_loss_lst.append(avg_train_loss)
        avg_val_loss_lst.append(avg_val_loss)

        if early_stopper.early_stop(avg_val_loss):
            break
        if (epoch + 1) % 20 == 0:
            print(f"Epoch = {epoch + 1}, Train Loss = {avg_train_loss:.4f}, Validation Loss = {avg_val_loss:.4f}")

    # plot the learning curve
    epochs_range = [i for i in range(len(avg_train_loss_lst))]
    plt.plot(epochs_range, avg_train_loss_lst, label="Train")
    plt.plot(epochs_range, avg_val_loss_lst, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def model_by_model_type(model_type, n_features, cfg):
    """
    Returns a model instance based on the model type, with configurations provided in a dictionary.

    Args:
        model_type (str): Type of the model to be instantiated ('LSTM', 'FCNN', 'LSTM_BNN').
        n_features (int): Number of features in the input data.
        cfg (dict): An object containing model parameters.

    Raises:
        RuntimeError: If `model_type` is not supported.

    Returns:
        An instance of the specified model, configured according to `cfg`.
    """
    if model_type == 'LSTM':
        num_layers = getattr(cfg, 'num_layers', 1)
        hidden1 = getattr(cfg, 'hidden1', 32)
        hidden2 = getattr(cfg, 'hidden2', 32)
        model = LSTM(input_size=n_features, num_layers=num_layers,
                     hidden1=hidden1, hidden2=hidden2,
                     device=getattr(cfg, 'device', 'cpu')).to(getattr(cfg, 'device', 'cpu'))
    else:
        raise RuntimeError(f"Unsupported model type: '{model_type}'.")

    return model


class NetWork:
    # accept either the dataframes or preprocessed datasets as initialization
    def __init__(self, cfg, model_type, save_model_name,
                 train_step_f, val_step_f, predict_step_f,
                 train_dataset, test_dataset,
                 save_dir='../result/24-01-12-FD003',
                 loss_fn_regression=None):

        self.cfg = cfg
        self.model_type = model_type
        self.save_model_name = save_model_name

        self.train_step_f = train_step_f
        self.val_step_f = val_step_f
        self.predict_step_f = predict_step_f

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.save_dir = os.path.join(save_dir, model_type)
        os.makedirs(self.save_dir, exist_ok=True)

        self.loss_fn_regression = loss_fn_regression

        # train model
        self.model = self.train_network()

    def train_network(self):
        """
        Sets up the datasets and loaders, initializes and trains the model, and saves the trained model.
        """
        self.features = self.train_dataset.features
        self.n_features = self.train_dataset.n_features

        # train_loader
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.cfg.batch_size,
                                                        shuffle=True)  # Training data is shuffled for better
        # generalization
        self.train_loader_no_shuffle = torch.utils.data.DataLoader(self.train_dataset,
                                                                   batch_size=self.cfg.batch_size,
                                                                   shuffle=False)
        # test_loader
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                       batch_size=self.cfg.batch_size,
                                                       shuffle=False)

        # Initialize the model and optimizer
        model = model_by_model_type(self.model_type, self.n_features, self.cfg)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.lr)

        # Run the training process
        run(self.train_loader, self.test_loader,
            self.train_step_f, self.val_step_f,
            model,
            self.cfg.epochs, self.cfg.device, self.loss_fn_regression, optimizer)

        # Save the trained model
        model_path = os.path.join(self.save_dir, self.save_model_name)
        torch.save(model.state_dict(), model_path)

        return model

    def get_prediction_one(self, instance):
        """
        Makes a prediction for a single instance using the trained model.
        
        Args:
            instance (Tensor): The input instance for prediction, excluding the batch dimension.

        Returns:
            Tensor: The predicted mean if flag_uq is False, or a tuple (mean, std) if flag_uq is True.
        """
        # Ensure the instance has the correct shape (1, window_size, n_features)
        instance = instance.unsqueeze(0)  # Add a new axis to simulate a batch size of 1
        instance = instance.to(self.cfg.device)  # Move instance to the configured device
        self.model.eval()
        with torch.no_grad():
            mean = self.model(instance)
            return mean

    def save_prediction_result(self, flag_test=True):
        """
        Processes and saves the prediction results, calculates and prints evaluation metrics.
        
        Args:
            flag_test (bool): Determines whether to process test dataset predictions; if False, process training dataset.
        """

        # Determine which dataset to process based on flag_test
        key = 'test' if flag_test else 'train'
        loader = self.test_loader if flag_test else self.train_loader_no_shuffule
        df = self.predict_step_f(loader, self.cfg.device, self.model)

        final_result = pd.DataFrame()
        rul_mean = self.train_dataset.mean['RUL']
        rul_std = self.train_dataset.std['RUL']

        if self.model_type in ['LSTM']:
            final_result = df.copy()
            final_result["pred_rul"] = final_result["pred_rul"].apply(lambda x: x * rul_std + rul_mean)
            final_result["true_rul"] = final_result["true_rul"].apply(lambda x: x * rul_std + rul_mean)

            # get score                
            rmse = rmse_score(final_result['pred_rul'], final_result['true_rul'])
            mape = mape_score(final_result['pred_rul'], final_result['true_rul'])
            mae = mae_score(final_result['pred_rul'], final_result['true_rul'])

            print(f"{rmse = }, {mape = }, {mae = }")
            print()
        else:
            raise RuntimeError('Not Valid')

        file_name = f'result-{self.model_type}-{key}.csv'
        final_result.to_csv(os.path.join(self.save_dir, file_name), index=False)

        # Memory management
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rmse, mape, mae

    def plot_pred_unit(self, df, unit_id):
        """
        Plot the result of LSTM or FCNN model for a specific unit.
    
        Args:
            df (pd.DataFrame): DataFrame containing the predictions and true RUL values.
            unit_id (int): The specific unit_id to plot predictions for.
        """

        unit_df = df.query(f'unit_id == {unit_id}').reset_index(drop=True)
        c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        cycle_time = [i for i in range(len(unit_df))]

        plt.figure(figsize=(12, 9))
        plt.plot(cycle_time, unit_df['pred_rul'], color=c[3], label='prediction')  # predicted avg
        plt.plot(cycle_time, unit_df['true_rul'], color='black', label='true')  # ground truth

        plt.xlabel('Cycle Time', fontsize=FONT_SIZES["default"])
        plt.ylabel('RUL', fontsize=FONT_SIZES["default"])
        plt.title(f'test unit: {unit_id}')
        plt.legend()
        plt.show()


def run_exp(cfg, model_type, train_dataset, test_dataset,
            save_dir='../result/24-01-12'):
    os.makedirs(save_dir, exist_ok=True)

    print(f'---------{model_type = }----------')

    # Assign functions and parameters based on model_type
    if model_type in ['LSTM']:
        train_step_f = train_step
        val_step_f = val_step
        predict_step_f = predict_step
        loss_fn_regression = torch.nn.MSELoss(reduction='mean')
    else:
        raise RuntimeError(f'{model_type} does not exist')

    save_model_name = model_type + ".pth"

    net = NetWork(cfg, model_type, save_model_name,
                  train_step_f, val_step_f, predict_step_f,
                  train_dataset, test_dataset,
                  save_dir, loss_fn_regression)

    return net


def sensor_selction_static(ds_name, full=False):
    cfg = config.Config_basic()
    cfg.lr = 0.0001
    cfg.batch_size = 128
    cfg.num_layers = 2
    cfg.hidden1 = 32
    cfg.hidden2 = 16
    cfg.window_size = 60
    cfg.epochs = 100

    for seed in range(10):
        set_seed(seed)
        ds_dir = '../dataset/Aircraft Engine/CMaps'
        df_train, df_test = prepare_data(ds_dir=ds_dir,
                                         ds_name=ds_name,
                                         extract_rul_method='linear',
                                         drop_useless=True,
                                         drop_feature_lst=[])
        df_test['id'] = df_test['id'].astype(int)
        df_train['id'] = df_train['id'].astype(int)
        cols = df_train.columns
        df_test_copy = df_test.copy()

        # fill non-transformed columns with overall mean
        if ds_name in ['FD001', 'FD003']:
            if not full:
                trans_cols = ['id', 'cycle', 'T24', 'T50', 'P30', 'Nf', 'Ps30', 'phi',
                              'NRf', 'BPR', 'htBleed', 'W31', 'W32', 'RUL', 'WC']
            else:
                trans_cols = list(cols)   # Use all columns

            non_trans_cols = list(set(cols) - set(trans_cols))
            for c in non_trans_cols:
                df_test_copy[c] = df_train[c].mean()

        else:
            # fill non-transformed columns with mean grouped by 'WC'
            if not full:
                trans_cols = ['id', 'cycle', 'Altitude', 'Mach_number', 'TRA',
                              'T24', 'T50', 'P30', 'Nf', 'Ps30', 'phi',
                              'NRf', 'BPR', 'htBleed', 'W31', 'W32', 'RUL', 'WC']
            else:
                trans_cols = list(cols)

            non_trans_cols = list(set(cols) - set(trans_cols))
            print(f'{non_trans_cols = }')
            for c in non_trans_cols:
                # For each unique WC value in df_test, fill with mean of corresponding WC in df_train
                for wc in df_test_copy['WC'].unique():
                    mask = df_test_copy['WC'] == wc
                    df_test_copy.loc[mask, c] = df_train[df_train['WC'] == wc][c].mean()

        train_dataset = SequenceDataset(df_train, ds_name=ds_name,
                                        window_size=cfg.window_size)
        test_dataset = SequenceDataset(df_test_copy, ds_name=ds_name,
                                       window_size=cfg.window_size,
                                       mean=train_dataset.mean, std=train_dataset.std)

        save_path = f'../result/25-01-24-{ds_name}-static' if not full else f'../result/25-01-24-{ds_name}-full'
        net = run_exp(cfg=cfg, model_type='LSTM',
                      train_dataset=train_dataset, test_dataset=test_dataset,
                      save_dir=os.path.join(save_path, f'seed-{seed}'))
        net.save_prediction_result(flag_test=True)


if __name__ == "__main__":
    for ds_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        sensor_selction_static(ds_name='FD001', full=True)   # full
        sensor_selction_static(ds_name='FD001', full=False)  # static

