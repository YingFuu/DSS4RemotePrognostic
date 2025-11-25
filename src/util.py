# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 16:38:57 2023

@author: Ying Fu
"""

import numpy as np
import pandas as pd
import torch


def tensor2numpy(tensor_v):
    """
    Convert a PyTorch tensor to a NumPy array or return the input if it's already a NumPy array or a native Python numeric type.
    """
    if tensor_v is None:
        return None

    if isinstance(tensor_v, np.ndarray):
        return tensor_v

    if isinstance(tensor_v, torch.Tensor):
        # Check if the tensor is stored on the GPU and move it to the CPU
        if tensor_v.is_cuda:
            tensor_v = tensor_v.cpu()
        return tensor_v.detach().numpy()

    if isinstance(tensor_v, (int, float, complex)):
        return tensor_v

    raise TypeError("Input must be a PyTorch tensor, NumPy array, or a native Python numeric type.")


class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.1):
        self.patience = patience  # number of epochs to wait after the last improvement before stopping the training
        self.min_delta = min_delta  # Minimum change in the monitored quantity to qualify as an improvement.
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Determine whether the training process should stop early
        Watch for the trend in validation loss alone,
            i.e., if the training is not resulting in lowering of the
            validation loss then terminate it.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def rmse_score(pred, true):
    """
    Root Mean Square Error
    RMSE = \sqrt {\sum _{i=1}^n(pred_i- true_i)^2 / n }
    """
    if len(pred) != len(true):
        raise RuntimeError

    if len(pred) == 0:
        return None

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    return round(np.sqrt(((pred - true) ** 2).mean()), 4)


def mae_score(pred, true):
    """
    Mean absolute error
    """
    if len(pred) != len(true):
        raise RuntimeError

    if len(pred) == 0:
        return None

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    return abs(pred - true).mean()


def mape_score(pred, true):
    """
    Mean Absolute Percentage Error
    MAPE = {\sum_{i=1}^{n} |pred_t-true_t|/true_t} / n
    """
    if len(pred) != len(true):
        raise RuntimeError

    if len(pred) == 0:
        return None

    if not isinstance(pred, np.ndarray):
        pred = np.array(pred)
    if not isinstance(true, np.ndarray):
        true = np.array(true)

    epsilon = 1e-12
    return round(np.abs((pred - true) / (true + epsilon)).mean(), 4)


def append_row(df, row):
    df = df.copy()
    df.loc[len(df)] = row
    return df



def random_selection_df(df, start_cycle, ratio_sensors_to_mask, ds_name):
    """
    Randomly selects a specified number of sensor columns to keep their current values,
    and applies a forward fill to the remaining sensor values from the specified start cycle.

    Parameters:
        df (pd.DataFrame): The input dataframe with a 'cycle' column.
        start_cycle (int): The cycle number from which to start modifications.
        ratio_sensors_to_mask (float): The ratio of sensor columns to randomly keep without modification.

    Returns:
        pd.DataFrame: The modified DataFrame.
    """
    all_columns = list(df.columns)
    non_sensor_columns = ['id', 'cycle', 'Altitude', 'Mach_number', 'TRA', 'RUL', 'WC']
    sensor_columns = [col for col in all_columns if col not in non_sensor_columns]
    num_sensors_to_mask = int(len(sensor_columns) * ratio_sensors_to_mask)
    if num_sensors_to_mask > len(sensor_columns):
        raise ValueError("num_sensors_to_mask exceeds the number of available sensor columns")

    if num_sensors_to_mask < 1:
        raise ValueError("num_sensors_to_mask must be at least 1")

    def process_unit(unit_df):
        df_to_modify = unit_df[unit_df['cycle'] > start_cycle]
        mask_array = np.zeros((len(df_to_modify), len(sensor_columns)), dtype=bool)
        for i in range(len(df_to_modify)):
            mask_indices = np.random.choice(len(sensor_columns), size=num_sensors_to_mask, replace=False)
            mask_array[i, mask_indices] = True

        modified_index = unit_df['cycle'] > start_cycle
        unit_df.loc[modified_index, sensor_columns] = df_to_modify[sensor_columns].where(mask_array, np.nan)

        print(f'{unit_df = }')

        if ds_name in ['FD001', 'FD003']:
            unit_df = unit_df.ffill()
        elif ds_name in ['FD002', 'FD004']:
            # Forward fill within the same working condition
            unit_df = unit_df.groupby('WC', group_keys=False).apply(lambda group: group.ffill())
        else:
            raise RuntimeError(f'{ds_name} does not exist.')
        return unit_df

    df = df.groupby('id', group_keys=False).apply(process_unit)

    return df


def compare_2_dfs(df1, df2):
    """
    Check whether two dataframes are identical. If not, identify where the values differ
    and stores this information into a new dataframe
    """
    are_equal = df1.equals(df2)
    if are_equal:
        print("The DataFrames are exactly the same.")
    else:
        print("The DataFrames are not the same.")
        comparison = df1 == df2
        differences = np.where(comparison == False)
        diff_locations = pd.DataFrame({
            'Row': differences[0],
            'Column': differences[1],
            'df1_value': df1.values[differences],
            'df2_value': df2.values[differences]
        })
        print(diff_locations)
