# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:55:51 2024

@author: Ying Fu
"""

import os
import time
import random
import math
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from matplotlib import rc

# Custom module imports
from src.data.JetEngine import prepare_data
from dataloader.sequence_dataloader import SequenceDataset
import config
from util import rmse_score, append_row, mape_score, mae_score, random_selection_df, tensor2numpy
from baseline_exp import set_seed, run_exp
from local_linear_approximator import LinearApproximator
from subset_closest_sum_solver import DPsolver

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# setting for plots
FONT_SIZES = {
    "default": 36,
    "legend": 24,
}
plt.rc('font', size=FONT_SIZES["default"])  # Default text size
plt.rc('axes', titlesize=FONT_SIZES["default"])  # Axes title font size
plt.rc('axes', labelsize=FONT_SIZES["default"])  # Axes label font size
plt.rc('xtick', labelsize=FONT_SIZES["default"])  # X-tick label size
plt.rc('ytick', labelsize=FONT_SIZES["default"])  # Y-tick label size
plt.rc('legend', fontsize=FONT_SIZES["legend"])  # Legend font size
plt.rc('figure', titlesize=FONT_SIZES["default"])  # Figure title font size


def _metric_func(full_pred_mean, partial_pred_mean):
    """
    Squared performance difference between prediction using all sensor data and partial sensor data. 
    A lower value is preferred.        
    """

    return torch.sqrt((full_pred_mean - partial_pred_mean) ** 2)


def _get_new_record_by_selected_index(selected_sensor_lst,
                                      previous_record, current_record):
    """
    Generate a new record by selecting values from the current and previous records
    based on the selected sensors. 
    
    Parameters:
    - selected_sensor_lst: Indices of selected sensors.
    - previous_record: current time step sensor records.
    - current_record: current time step sensor records.
    
    Returns:
    - A new record based on the selection.
    """

    if not (torch.is_tensor(previous_record) and torch.is_tensor(current_record)):
        raise ValueError("Input records must be PyTorch tensors")

    assert torch.is_tensor(selected_sensor_lst), "selected_sensor_lst must be a tensor of selected indices."
    selected_sensor_lst = selected_sensor_lst.long()  # Ensure that the indices are of integer type

    new_record = previous_record.clone()
    new_record[0, selected_sensor_lst] = current_record[
        0, selected_sensor_lst]  # tensors used as indices must be long, byte or bool tensors

    return new_record


def _find_similar_wc(current_wc, nwt_history):
    '''
    Define a dictionary of similarities for working conditions and use it 
    to find the closest matching working condition.
    '''

    similarity_dict = {
        0: [1],
        1: [2],
        2: [1],
        3: [4],
        4: [3],
        5: [3, 4]}

    if current_wc not in similarity_dict:
        return None

    similar_wcs = similarity_dict[current_wc]
    for wc in similar_wcs:
        same_wc_indices = (nwt_history[:, -1] == wc).nonzero(as_tuple=True)[0]
        if same_wc_indices.numel() > 0:
            latest_same_wc_index = same_wc_indices[-1]
            return nwt_history[latest_same_wc_index, :].unsqueeze(0)

    return None


def _unit_selection_heatmap_plot(df, unit_id, save_path):
    """
    Plots a heatmap for a given matrix with columns representing sensors and rows representing cycle times.
    The color represents the activation of each sensor across cycles.
    
    Parameters:
    - array: A numpy array with sensor data across different cycles.
    - column_names: A list of names corresponding to each sensor column in the array.
    - colormap: The colormap to use for the heatmap.
    """

    column_sums = np.sum(df, axis=0)
    x_tick_labels = [f'{name} ({sum:.0f})' for name, sum in zip(df.columns, column_sums)]

    plt.figure(figsize=(12, 9))
    sns.heatmap(df, annot=False, cmap="YlGnBu", linewidth=1, cbar=False)
    plt.title(f' unit id {unit_id}')
    plt.xlabel('Sensors', fontsize=FONT_SIZES["default"])
    plt.ylabel('Cycle Time', fontsize=FONT_SIZES["default"])
    plt.xticks(ticks=np.arange(len(x_tick_labels)) + 0.5, labels=x_tick_labels, rotation=60, ha='right')
    plt.title(f'unit id: {unit_id}')
    plt.savefig(os.path.join(save_path, f'unit {unit_id}.pdf'), bbox_inches='tight')
    # plt.savefig(os.path.join(save_path, f'unit {unit_id}.svg'), bbox_inches='tight')
    # plt.show()
    plt.close()


class Dynamic_sensor_selection:
    def __init__(self, ds_name, net, ntw, n_sensors,
                 sel_ratio=None, sel_sensor_num=None, flag_uq=False):

        self.ds_name = ds_name
        self.flag_uq = flag_uq

        self.net = net  # network model
        self.train_dataset = self.net.train_dataset  # the current dataset used for training net, we need to update it
        self.test_dataset = self.net.test_dataset
        self.rul_mean = self.train_dataset.mean['RUL']
        self.rul_std = self.train_dataset.std['RUL']

        self.ntw = ntw  # time window size

        self.n_sensors = n_sensors  # number of total sensors
        if sel_sensor_num is not None:
            self.sel_sensor_num = sel_sensor_num
        elif sel_ratio is not None and 0 < sel_ratio < 1:
            self.sel_sensor_num = int(n_sensors * sel_ratio)
        else:
            raise ValueError("Either 'sel_sensor_num' or both 'n_sensors' and 'sel_ratio' must be provided.")

        if self.sel_sensor_num < 1:
            raise RuntimeError(
                f"The number of selected sensors ({self.sel_sensor_num}) must be greater than or equal to 1.")

        print(f'selected sensors: {self.sel_sensor_num}, total number of sensors: {n_sensors}')

    def get_previous_record(self, previous_sent_instance, current_record):

        nwt_history = previous_sent_instance[:-1, :]  # History without the current record, size: (ntw-1, features)

        if self.ds_name in ['FD001', 'FD003']:
            previous_record = previous_sent_instance[-1, :].unsqueeze(
                0)  # Last cycle time the record that sent back to server, size: (1, features)
        elif self.ds_name in ['FD002', 'FD004']:
            current_wc = current_record[-1, -1].item()  # Get the working condition of the current record, a scalar 
            # Find the latest record with the same working condition
            same_wc_indices = (nwt_history[:, -1] == current_wc).nonzero(as_tuple=True)[0]
            if same_wc_indices.numel() == 0:
                # Handle case when no previous record with the same working condition is found
                print("No previous record with the same working condition found. Searching for similar working "
                      "conditions.")
                previous_record = _find_similar_wc(current_wc, nwt_history)
                if previous_record is None:
                    previous_record = previous_sent_instance[-1, :].unsqueeze(0)  # Fallback to the most recent record
                    print("No similar working condition found. Using the most recent record.")
            else:
                latest_same_wc_index = same_wc_indices[-1]
                previous_record = nwt_history[latest_same_wc_index, :].unsqueeze(0)
        else:
            raise ValueError(f"Dataset name {self.ds_name} is not recognized.")
        return previous_record

    def random_selection(self, previous_sent_instance, current_record):
        """
        Dynamically selects sensors at each time step in a random manner. If a random seed is provided, 
        it ensures the fixed selection.        
        """

        nwt_history = previous_sent_instance[:-1, :]  # History without the current record, size: (ntw-1, features)
        previous_record = self.get_previous_record(previous_sent_instance, current_record)

        # Select unique random sensors
        F = torch.randperm(self.n_sensors)[:self.sel_sensor_num]
        server_record = _get_new_record_by_selected_index(F, previous_record, current_record)
        server_instance = torch.cat((nwt_history, server_record), dim=0)

        return F, server_record, server_instance

    def greedy_selection(self, previous_sent_instance, current_record):
        """
        select sensors greedily based on a metric that compares partial predictions 
        (with some sensors selected) against full predictions (using all sensors)
        """

        if not all(torch.is_tensor(x) for x in [previous_sent_instance, current_record]):
            raise ValueError("All input records must be PyTorch tensors")

        nwt_history = previous_sent_instance[:-1, :]  # size: [# time step, # features]
        previous_record = self.get_previous_record(previous_sent_instance, current_record)

        current_edge_instance = torch.cat((nwt_history, current_record), dim=0)
        full_pred_mean = self.net.get_prediction_one(current_edge_instance)

        F = torch.tensor([], dtype=torch.long)  # Indices of selected sensors
        while F.numel() < self.sel_sensor_num:
            best_i = None
            best_metric = math.inf
            for i in range(self.n_sensors):
                if i not in F:
                    F_i = torch.cat((F, torch.tensor([i])))
                    # partial prediction
                    new_record = _get_new_record_by_selected_index(F_i, previous_record, current_record)
                    current_server_instance = torch.cat((nwt_history, new_record), dim=0)
                    partial_pred_mean = self.net.get_prediction_one(current_server_instance)

                    current_metric = _metric_func(full_pred_mean, partial_pred_mean)
                    if current_metric < best_metric:
                        best_metric = current_metric
                        best_i = i

            if best_i is None:
                raise ValueError("No suitable sensor found to add; ensure n_sensors and sel_ratio are set correctly.")
            F = torch.cat((F, torch.tensor([best_i])))

        server_record = _get_new_record_by_selected_index(F, previous_record, current_record)
        server_instance = torch.cat((nwt_history, server_record), dim=0)
        return F, server_record, server_instance

    def genetic_selection(self, previous_sent_instance, current_record):
        if not all(torch.is_tensor(x) for x in [previous_sent_instance, current_record]):
            raise ValueError("All input records must be PyTorch tensors")

        nwt_history = previous_sent_instance[:-1, :]  # size: [# time step, # features]
        previous_record = self.get_previous_record(previous_sent_instance, current_record)
        current_edge_instance = torch.cat((nwt_history, current_record), dim=0)
        full_pred_mean = self.net.get_prediction_one(current_edge_instance)

        def fitness(individual):
            selected_indices = torch.tensor([i for i, v in enumerate(individual) if v == 1])
            new_record = _get_new_record_by_selected_index(selected_indices, previous_record, current_record)
            current_server_instance = torch.cat((nwt_history, new_record), dim=0)
            partial_pred_mean = self.net.get_prediction_one(current_server_instance)
            return _metric_func(full_pred_mean, partial_pred_mean)

        # --- Initialize population ---
        def initialize_population(pop_size, n_sensors, k):
            population = []
            for _ in range(pop_size):
                individual = [0] * n_sensors
                selected = np.random.choice(n_sensors, k, replace=False)
                for idx in selected:
                    individual[idx] = 1
                population.append(individual)
            return population

        def repair(individual, k):
            ones = [i for i, bit in enumerate(individual) if bit == 1]
            zeros = [i for i, bit in enumerate(individual) if bit == 0]
            if len(ones) > k:
                flip_off = np.random.choice(ones, len(ones) - k, replace=False)
                for i in flip_off:
                    individual[i] = 0
            elif len(ones) < k:
                flip_on = np.random.choice(zeros, k - len(ones), replace=False)
                for i in flip_on:
                    individual[i] = 1
            return individual

        def fixed_k_mutation(individual, k, mutation_rate):
            if np.random.rand() < mutation_rate:
                ones = [i for i, bit in enumerate(individual) if bit == 1]
                zeros = [i for i, bit in enumerate(individual) if bit == 0]
                if ones and zeros:
                    off = np.random.choice(ones)
                    on = np.random.choice(zeros)
                    individual[off] = 0
                    individual[on] = 1
            return individual

        generations = 15
        pop_size = int(0.8 * self.n_sensors)
        mutation_rate = 0.1
        k = self.sel_sensor_num
        population = initialize_population(pop_size, self.n_sensors, k)

        for _ in range(generations):
            scores = [fitness(ind).item() for ind in population]
            sorted_idx = np.argsort(scores)
            population = [population[i] for i in sorted_idx]
            next_gen = population[:5]  # elitism

            while len(next_gen) < pop_size:
                parent_pool_size = min(10, len(population))
                idx1, idx2 = np.random.choice(parent_pool_size, 2, replace=False)
                p1 = population[idx1]
                p2 = population[idx2]
                point = np.random.randint(1, self.n_sensors - 1)
                c1 = p1[:point] + p2[point:]
                c2 = p2[:point] + p1[point:]

                for child in [c1, c2]:
                    child = repair(child, k)
                    child = fixed_k_mutation(child, k, mutation_rate)
                    next_gen.append(child)
                    if len(next_gen) >= pop_size:
                        break

            population = next_gen

        # --- Final output ---
        best_individual = population[0]
        F = torch.tensor([i for i, v in enumerate(best_individual) if v == 1], dtype=torch.long)

        server_record = _get_new_record_by_selected_index(F, previous_record, current_record)
        server_instance = torch.cat((nwt_history, server_record), dim=0)
        return F, server_record, server_instance

    def linear_approximation_selection(self, previous_sent_instance, current_record,
                                       num_samples, noise_std, scale_factor):
        """
        Perform a local linear approximation to the current model.
        Previously using sklearn to do the linear regression. However, it requires numerous 
        conversion between PyTorch tensors and NumPy arrays. 
        """
        nwt_history = previous_sent_instance[:-1, :]  # Size: [# time step, # features]
        previous_record = self.get_previous_record(previous_sent_instance, current_record)

        LR_approximator = LinearApproximator(num_samples, noise_std, scale_factor)
        X, y, weights = LR_approximator.generate_samples(nwt_history, current_record, self.net)
        beta = LR_approximator.perform_linear_regression(X, y, weights)

        slopes = beta[1:]

        gap = (current_record - previous_record).squeeze(0)  # [1, num_features] --> [num_features]
        impact_scores = gap * slopes  # sum_i wi (xi_{t-1} - xi_{t}) \approx 0

        seq_len = self.net.n_features - self.sel_sensor_num
        dp_solver = DPsolver(seq_len, 0, impact_scores, 0)
        (gap, sel_indices) = dp_solver.subset_sum_dp_discrete()
        indices_all = torch.arange(self.net.n_features)
        mask = ~torch.isin(indices_all, sel_indices)
        F = indices_all[mask]  # a tensor list

        # Construct the final server record and instance after selecting sensors
        server_record = _get_new_record_by_selected_index(F, previous_record, current_record)
        server_instance = torch.cat((nwt_history, server_record), dim=0)

        return F, server_record, server_instance

    def selection_for_one_unit(self, ds_name, df_unit, unit_id, sel_method,
                               num_samples, noise_std, scale_factor,
                               sel_train_flag=False, save_dir=None):

        """
        dynamic sensor selection and RUL prediction for a specific unit in a dataset.
        """

        unit_dataset = SequenceDataset(df_unit, ds_name=ds_name,
                                       window_size=cfg.window_size,
                                       mean=self.train_dataset.mean, std=self.train_dataset.std)
        new_dataset_tuples = [unit_dataset[0]]
        center_receive = unit_dataset[0][0]

        df = pd.DataFrame(columns=['unit_id', 'cycle', 'sensor_index_lst',
                                   'sensor_index_array', 'sensor_cols',
                                   'pred_rul_partial',
                                   'true'])

        for i in range(len(unit_dataset) - 1):
            previous_sent_instance = center_receive[i: i + cfg.window_size,
                                     :]  # from central server, size: [ntw, feature]
            current_instance = unit_dataset[i + 1][0]  # from the unit, size: [ntw-1, feature]
            current_record = current_instance[-1, :].unsqueeze(0)  # size: [1, feature]

            # Select sensors based on the specified selection method
            if sel_method == 'greedy':
                F, server_record, server_instance = self.greedy_selection(previous_sent_instance, current_record)
            elif sel_method == 'genetic':
                F, server_record, server_instance = self.genetic_selection(previous_sent_instance, current_record)
            elif sel_method == 'linear_approx':
                F, server_record, server_instance = self.linear_approximation_selection(previous_sent_instance,
                                                                                        current_record,
                                                                                        num_samples, noise_std,
                                                                                        scale_factor)
            elif sel_method == 'random':
                F, server_record, server_instance = self.random_selection(previous_sent_instance, current_record)
            else:
                raise RuntimeError("Invalid selection method specified.")

            new_data = (server_instance, unit_dataset[i + 1][1], unit_dataset[i + 1][2])
            new_dataset_tuples.append(new_data)

            # Update the center's data with the new record
            center_receive = torch.cat((center_receive, server_record), dim=0)

            pred_rul_partial = self.net.get_prediction_one(server_instance)
            pred_rul_partial = pred_rul_partial.item() * self.rul_std + self.rul_mean
            pred_rul_partial = tensor2numpy(pred_rul_partial)

            true = unit_dataset[i + 1][1].item() * self.rul_std + self.rul_mean

            A = torch.zeros(self.n_sensors)
            A[F] = 1

            new_row = pd.Series({'unit_id': unit_id, 'cycle': i + cfg.window_size + 1,
                                 'sensor_index_lst': tensor2numpy(F),
                                 'sensor_index_array': tensor2numpy(A),
                                 'sensor_cols': self.net.features,
                                 'pred_rul_partial': pred_rul_partial,
                                 'true': true})
            df = append_row(df, new_row)

        rmse = rmse_score(df['pred_rul_partial'], df['true'])

        if sel_train_flag:
            self.train_dataset.add_new_samples(new_dataset_tuples)

        sel_sensor_matrix = np.array(df['sensor_index_array'].tolist())
        sel_sensor_matrix = pd.DataFrame(sel_sensor_matrix)
        sel_sensor_matrix.columns = self.net.features
        sel_sensor_matrix.index = df['cycle']
        column_sums = np.sum(sel_sensor_matrix, axis=0)
        _unit_selection_heatmap_plot(sel_sensor_matrix, unit_id, save_dir)

        return df, rmse, column_sums

    def sensor_select_for_units(self, ds_name, df, sel_train_flag,
                                sel_method, num_samples, noise_std,
                                save_dir='../result/24-04-06'):

        unit_id_lst = df['id'].unique()

        start_time = time.process_time()
        select_count_stat_dict = {}  # records the sum of each selected sensors for each each unit
        df_result = []
        print(f'---{sel_method = }----')
        for unit_id in unit_id_lst:
            print(f'  {unit_id = }')
            df_unit = df.query(f'id == {unit_id}').reset_index(drop=True)
            if len(df_unit) > self.ntw:
                sel_result_df, rmse_partial, column_sums = self.selection_for_one_unit(ds_name, df_unit, unit_id,
                                                                                       sel_method,
                                                                                       num_samples, noise_std,
                                                                                       scale_factor,
                                                                                       sel_train_flag=sel_train_flag,
                                                                                       save_dir=save_dir)
                df_result.append(sel_result_df)
                select_count_stat_dict[
                    unit_id] = column_sums  # PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.

        df_result = pd.concat(df_result)
        df_result.to_csv(os.path.join(save_dir, 'result.csv'), index=False)

        rmse = rmse_score(df_result['pred_rul_partial'], df_result['true'])
        mape = mape_score(df_result['pred_rul_partial'], df_result['true'])
        mae = mae_score(df_result['pred_rul_partial'], df_result['true'])
        elapsed_time = time.process_time() - start_time
        print(f'selected_sensors_num: {self.sel_sensor_num}, method: {sel_method}, \
              {rmse = }, {mape = }, {mae =}, time = {elapsed_time} s')

        return rmse, mape, mae, elapsed_time, self.sel_sensor_num


def sensor_selection_iterative_updating_exp(ds_name, df_train, df_test,
                                            num_samples, noise_std,
                                            initial_units_nums=10, update_period=20,
                                            sel_method='greedy', sel_ratio=0.1,
                                            save_dir=None):
    if initial_units_nums == 0:
        cold_start = True
    else:
        cold_start = False

    unit_idx = list(df_train['id'].unique())
    total_units_nums = len(unit_idx)

    if initial_units_nums == total_units_nums:
        save_dir = os.path.join(save_dir,
                                f'full_historical-sel_method-{sel_method}-sel_ratio-{sel_ratio}')
    else:
        save_dir = os.path.join(save_dir,
                                f'initial_num-{initial_units_nums}-period-{update_period}-sel_method-{sel_method}-sel_ratio-{sel_ratio}')

    net_performance_df = pd.DataFrame(columns=['total_units_nums', 'initial_units_nums',
                                               'update_period', 'update_times',
                                               'rmse', 'mape', 'mae',
                                               'sel_method', 'sel_ratio'])

    initial_units_nums = update_period if initial_units_nums == 0 else initial_units_nums  # to combine the case of cold start + partial start

    initial_units_idxs = random.sample(unit_idx, initial_units_nums)
    df_train_initial = df_train[df_train['id'].isin(initial_units_idxs)].reset_index(drop=True)
    df_train_remaining = df_train[~df_train['id'].isin(initial_units_idxs)].reset_index(drop=True)
    remaining_units_idxs = df_train_remaining['id'].unique()

    if cold_start:  # We only need to do random selection in cold start case.
        df_train_initial = random_selection_df(df_train_initial, cfg.window_size, sel_ratio, ds_name)

    train_dataset_initial = SequenceDataset(df_train_initial, ds_name=ds_name,
                                            window_size=cfg.window_size)
    test_dataset = SequenceDataset(df_test, ds_name=ds_name,
                                   window_size=cfg.window_size,
                                   mean=train_dataset_initial.mean,
                                   std=train_dataset_initial.std)

    # initial run to get net using initial training dataset
    update_times = 0
    net = run_exp(cfg=cfg, model_type='LSTM',
                  train_dataset=train_dataset_initial, test_dataset=test_dataset,
                  save_dir=os.path.join(save_dir, f'net-update-times-{update_times}'))
    rmse_net, mape_net, mae_net = net.save_prediction_result()
    new_row = pd.Series({'total_units_nums': total_units_nums,
                         'initial_units_nums': initial_units_nums,
                         'update_period': update_period,
                         'update_times': update_times,
                         'rmse': rmse_net,
                         'mape': mape_net,
                         'mae': mae_net,
                         'sel_method': sel_method,
                         'sel_ratio': sel_ratio})
    net_performance_df = append_row(net_performance_df, new_row)

    dss = Dynamic_sensor_selection(ds_name=ds_name, net=net, ntw=cfg.window_size,
                                   n_sensors=net.n_features,
                                   sel_ratio=sel_ratio)

    # not full selection case, we need select sensors for remaining training sets
    if initial_units_nums != total_units_nums:
        df_unit_lst = [df_train.query(f'id == {i}') for i in remaining_units_idxs]
        all_unit_dataset = []
        valid_remaining_units = []
        max_len_ds = 0

        for i, df_unit in enumerate(df_unit_lst):
            # Skip units that are too short to form even one window
            if len(df_unit) < cfg.window_size:
                print(f"Warning: Unit {remaining_units_idxs[i]} has only {len(df_unit)} samples, "
                      f"need at least {cfg.window_size}. Skipping...")
                continue

            unit_dataset = SequenceDataset(df_unit, ds_name=ds_name,
                                           window_size=cfg.window_size,
                                           mean=train_dataset_initial.mean,
                                           std=train_dataset_initial.std)

            # Double-check dataset is not empty after creation
            if len(unit_dataset) > 0:
                all_unit_dataset.append(unit_dataset)
                valid_remaining_units.append(remaining_units_idxs[i])
                if len(unit_dataset) > max_len_ds:
                    max_len_ds = len(unit_dataset)
            else:
                print(f"Warning: Unit {remaining_units_idxs[i]} dataset is empty after creation. Skipping...")

        # Update remaining_units_idxs to only include valid units
        remaining_units_idxs = valid_remaining_units

        all_center_receive = [all_unit_dataset[i][0][0] for i in range(len(all_unit_dataset))]
        new_dataset_tuples = [[all_unit_dataset[i][0]] for i in range(len(all_unit_dataset))]

        failed_status = [False] * len(all_unit_dataset)

        for i in range(max_len_ds - 1):
            for unit in range(len(failed_status)):
                if failed_status[unit]:
                    continue
                try:
                    center_receive = all_center_receive[unit]  # Record the server received data for each unit
                    unit_dataset = all_unit_dataset[unit]

                    if (i + cfg.window_size) > center_receive.size(0) or (i + 1) >= len(unit_dataset):
                        raise IndexError

                    previous_sent_instance = center_receive[i: i + cfg.window_size,
                                             :]  # From central server, size: [ntw, feature]
                    current_instance = unit_dataset[i + 1][0]  # From the unit, size: [ntw-1, feature]
                    current_record = current_instance[-1, :].unsqueeze(0)  # Size: [1, feature]

                    # Select sensors based on the specified selection method
                    if sel_method == 'greedy':
                        F, server_record, server_instance = dss.greedy_selection(previous_sent_instance, current_record)
                    elif sel_method == 'genetic':
                        F, server_record, server_instance = dss.genetic_selection(previous_sent_instance,
                                                                                  current_record)
                    elif sel_method == 'linear_approx':
                        F, server_record, server_instance = dss.linear_approximation_selection(previous_sent_instance,
                                                                                               current_record,
                                                                                               num_samples, noise_std,
                                                                                               scale_factor)
                    elif sel_method == 'random':
                        F, server_record, server_instance = dss.random_selection(previous_sent_instance, current_record)
                    else:
                        raise RuntimeError("Invalid selection method specified.")

                    # Update the center's data with the new record
                    center_receive = torch.cat((center_receive, server_record), dim=0)
                    all_center_receive[unit] = center_receive

                    new_data = (server_instance, unit_dataset[i + 1][1], unit_dataset[i + 1][2])
                    new_dataset_tuples[unit].append(new_data)

                except IndexError:
                    failed_status[unit] = True
                    dss.train_dataset.add_new_samples(
                        new_dataset_tuples[unit])  # once the unit is failed, add the new dataset to training dataset
                    if sum(failed_status) % update_period == 0 or sum(failed_status) == len(
                            failed_status):  # either reach the update period or all units are failed
                        # train the model again
                        update_times += 1
                        print(f'--------{update_times = }------')
                        net = run_exp(cfg=cfg, model_type='LSTM',
                                      train_dataset=dss.train_dataset, test_dataset=test_dataset,
                                      save_dir=os.path.join(save_dir, f'net-update-times-{update_times}'))
                        rmse_net, mape_net, mae_net = net.save_prediction_result()
                        new_row = pd.Series({'total_units_nums': total_units_nums,
                                             'initial_units_nums': initial_units_nums,
                                             'update_period': update_period,
                                             'update_times': update_times,
                                             'rmse': rmse_net,
                                             'mape': mape_net,
                                             'mae': mae_net,
                                             'sel_method': sel_method,
                                             'sel_ratio': sel_ratio})
                        net_performance_df = append_row(net_performance_df, new_row)
                        dss = Dynamic_sensor_selection(ds_name=ds_name, net=net, ntw=cfg.window_size,
                                                       n_sensors=net.n_features,
                                                       sel_ratio=sel_ratio)
                    continue

    net_performance_df.to_csv(os.path.join(save_dir, 'net_performance.csv'), index=False)

    # use the final dss for test datasets
    test_save_path = os.path.join(save_dir, 'train-False')
    os.makedirs(test_save_path, exist_ok=True)
    rmse, mape, mae, t, n_sel = dss.sensor_select_for_units(ds_name=ds_name, df=df_test, sel_train_flag=False,
                                                            sel_method=sel_method, num_samples=num_samples,
                                                            noise_std=noise_std,
                                                            save_dir=test_save_path)

    return rmse, mape, mae, t, n_sel


def exp_sensor_selection(num_samples, noise_std, scale_factor, seed_lst, ds_name,
                         method_lst,
                         sel_ratio_lst,
                         initial_units_nums_ratio_lst,
                         update_period_ratio_lst):
    for seed in seed_lst:
        set_seed(seed)
        ds_dir = '../dataset/Aircraft Engine/CMaps'
        df_train, df_test = prepare_data(ds_dir=ds_dir,
                                         ds_name=ds_name,
                                         extract_rul_method='linear',
                                         drop_useless=True,
                                         drop_feature_lst=[])
        unit_idx = list(df_train['id'].unique())
        total_train_units = len(unit_idx)

        # save_dir = f'../result/25-05-07-{ds_name}/scale_factor-{scale_factor}/seed-{seed}'
        save_dir = f'../result/25-01-04-{ds_name}/seed-{seed}'
        try:
            result = pd.read_csv(os.path.join(save_dir, 'summary.csv'))
        except:
            result = pd.DataFrame(columns=['seed',
                                           'total_train_units',
                                           'sel_ratio', 'sel_sensor_num',
                                           'method',
                                           'initial_units_nums_ratio', 'initial_units_nums',
                                           'update_period_ratio', 'update_period',
                                           'rmse_test', 'mape_test', 'mae_test',
                                           'time(s)'])
        for sel_ratio in sel_ratio_lst:
            for initial_units_nums_ratio in initial_units_nums_ratio_lst:
                initial_units_nums = int(initial_units_nums_ratio * total_train_units)
                for update_period_ratio in update_period_ratio_lst:
                    update_period = int(update_period_ratio * total_train_units)
                    if initial_units_nums_ratio + update_period_ratio > 1:
                        continue
                    if initial_units_nums_ratio < 1 and update_period_ratio == 0:
                        continue
                    for sel_method in method_lst:
                        set_seed(seed)
                        print('-------current config------')
                        print(f'{sel_ratio = }, {sel_method = }, {initial_units_nums = }, {update_period = }')
                        rmse, mape, mae, t, n_sel = sensor_selection_iterative_updating_exp(ds_name,
                                                                                            df_train, df_test,
                                                                                            num_samples, noise_std,
                                                                                            initial_units_nums=initial_units_nums,
                                                                                            update_period=update_period,
                                                                                            sel_method=sel_method,
                                                                                            sel_ratio=sel_ratio,
                                                                                            save_dir=save_dir)
                        new_row = pd.Series({'seed': seed,
                                             'total_train_units': total_train_units,
                                             'sel_ratio': sel_ratio, 'sel_sensor_num': n_sel,
                                             'method': sel_method,
                                             'initial_units_nums_ratio': initial_units_nums_ratio,
                                             'initial_units_nums': initial_units_nums,
                                             'update_period_ratio': update_period_ratio, 'update_period': update_period,
                                             'rmse_test': rmse, 'mape_test': mape, 'mae_test': mae,
                                             'time(s)': t})
                        result = append_row(result, new_row)
                        result.to_csv(os.path.join(save_dir, 'summary.csv'), index=False)


if __name__ == "__main__":
    # global variable    
    cfg = config.Config_basic()
    cfg.lr = 0.0001
    cfg.batch_size = 128
    cfg.num_layers = 2
    cfg.hidden1 = 32
    cfg.hidden2 = 16
    cfg.window_size = 60
    cfg.epochs = 100


    num_samples = 100
    noise_std = 0.05
    scale_factor = 2
    for ds_name in ['FD001', 'FD002', 'FD003', 'FD004']:
        exp_sensor_selection(method_lst=['greedy', 'genetic', 'linear_approx', 'random'],
                             seed_lst=[i for i in range(10)],
                             num_samples=num_samples,
                             noise_std=noise_std, scale_factor=scale_factor,
                             ds_name=ds_name, sel_ratio_lst=[0.1, 0.2, 0.4, 0.8],
                             initial_units_nums_ratio_lst=[1, 0.8, 0.4, 0.2, 0],
                             update_period_ratio_lst=[0.8, 0.4, 0.2, 0.1, 0])

    # LLA sensitity analysis
    # for num_samples in [50, 100, 200]:
    #     for noise_std in [0.0001, 0.01, 0.05, 0.4, 3]:
    #         exp_sensor_selection(method_lst = ['linear_approx'],
    #                              num_samples=num_samples, 
    #                              noise_std=noise_std, seed_lst=[i for i in range(10)],
    #                              ds_name=ds_name)
