# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 16:53:43 2025

@author: Ying Fu
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def _determine_rul_range(value):
    if value < 100:
        return r'RUL $< 100$'
    elif value <= 200:
        return r'$100 \leq$ RUL $\leq 200$'
    else:
        return r'RUL $> 200$'


def _determine_fm(value):
    # Fan 
    fm0_unit_id_lst = [1, 3, 5, 15, 16, 17, 18, 20, 21, 22, 23, 24, 27, 30, 35,
                       39, 40, 41, 43, 46, 49, 61, 62, 63, 64, 67, 68, 71, 72,
                       74, 75, 77, 78, 80, 81, 82, 85, 89, 92, 93, 94, 95, 96, 99, 100]
    # HPC
    fm1_unit_id_lst = [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 25, 26, 28, 29,
                       31, 32, 33, 34, 36, 37, 38, 42, 44, 45, 47, 48, 50, 51, 52, 53, 54, 55,
                       56, 57, 58, 59, 60, 65, 66, 69, 70, 73, 76, 79, 83, 84, 86, 87,
                       88, 90, 91, 97, 98]

    if value in fm0_unit_id_lst:
        return 'Fan Failure'
    elif value in fm1_unit_id_lst:
        return 'HPC Failure'
    else:
        raise ValueError(f"Unit ID {value} is not valid. Must be between 1 and 100.")


def _result_preprocess(df, ds_name):
    df['sensor_index_lst'] = df['sensor_index_lst'].apply(lambda x: list(map(int, x.strip('[]').split())))
    df['rul_range'] = df['pred_rul_partial'].apply(_determine_rul_range)
    if ds_name == 'FD003':
        df['fm'] = df['unit_id'].apply(_determine_fm)
    sensor_lst_str = df['sensor_cols'].iloc[0]
    sensor_lst = sensor_lst_str.strip('[]').replace("'", "").split(',')
    sensor_lst = [i.strip() for i in sensor_lst]
    df['selected'] = df['sensor_index_lst'].apply(lambda x: [sensor_lst[i] for i in x])

    return df


def plot_sensor_frequencies_no_group(ds_name, df_combined, seed_column='seed'):
    """
    Plot the sensor selection frequency. 
    """

    save_path = '../figures'
    os.makedirs(save_path, exist_ok=True)

    seed_results = []
    for seed, seed_group in df_combined.groupby(seed_column):
        all_sensors = [sensor for sensors_list in seed_group['selected'] for sensor in sensors_list]
        sensor_series = pd.Series(all_sensors)
        seed_df = sensor_series.value_counts(normalize=True).reset_index()
        seed_df.columns = ['sensor', 'frequency']
        seed_df['seed'] = seed
        seed_results.append(seed_df)

    all_seeds_df = pd.concat(seed_results)

    mean_df = all_seeds_df.groupby('sensor')['frequency'].mean()
    std_df = all_seeds_df.groupby('sensor')['frequency'].std()
    sorted_mean_df = mean_df.sort_values(ascending=False)
    sorted_std_df = std_df.reindex(sorted_mean_df.index)

    # plot
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_mean_df.index, sorted_mean_df.values,
            yerr=sorted_std_df.values,
            capsize=5, color="#999999", alpha=0.7)
    plt.ylabel('Frequency', fontsize=18)
    plt.xlabel('Sensor', fontsize=18)
    plt.title(f'{ds_name}')
    plt.xticks(rotation=75, fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'sensor_freq_{ds_name}.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'sensor_freq_{ds_name}.pdf'), bbox_inches='tight')
    plt.show()


def plot_sensor_frequencies_groups(df, group_name='rul_range', seed_column='seed'):
    """
    Count the frequency of selected sensors grouped by 'group_name' of sensor frequencies
    """
    save_path = '../figures'
    os.makedirs(save_path, exist_ok=True)

    if group_name == 'rul_range':
        new_order = [r'RUL $< 100$', r'$100 \leq$ RUL $\leq 200$', r'RUL $> 200$']
        df[group_name] = pd.Categorical(df[group_name], categories=new_order, ordered=True)
    seed_results = []
    for seed, seed_group in df.groupby(seed_column):
        results = []
        for g, group in seed_group.groupby(group_name, observed=False):
            all_sensors = [sensor for sensors_list in group['selected'] for sensor in sensors_list]
            sensor_series = pd.Series(all_sensors)
            sensor_counts = sensor_series.value_counts(normalize=True).to_dict()
            count_dict = {group_name: g}
            count_dict.update(sensor_counts)  # Insert items to the dictionary
            results.append(count_dict)
        seed_df = pd.DataFrame(results).fillna(0)
        seed_df = seed_df.set_index(group_name)
        seed_results.append(seed_df)

    all_seeds_df = pd.concat(seed_results, axis=0, keys=range(len(seed_results)),
                             names=['seed', group_name])

    mean_df = all_seeds_df.groupby(group_name).mean()
    std_df = all_seeds_df.groupby(group_name).std()

    if group_name == 'fm':
        selected_sensors = ['Nf', 'NRf', 'Ps30', 'P30']
        mean_df = mean_df.loc[:, selected_sensors]
        std_df = std_df.loc[:, selected_sensors]

    # sort by any of the group    
    transposed_mean_df = mean_df.T
    sorted_transposed_mean_df = transposed_mean_df.sort_values(by=transposed_mean_df.columns[0],
                                                               ascending=False)
    sorted_mean_df = sorted_transposed_mean_df.T
    # reindex the std df    
    transposed_std_df = std_df.T
    sorted_transposed_std_df = transposed_std_df.reindex(sorted_transposed_mean_df.index)
    sorted_std_df = sorted_transposed_std_df.T

    # plot
    sensors = sorted_mean_df.columns
    x = np.arange(len(sensors))
    levels = sorted_mean_df.index if group_name == 'fm' else [r'RUL $< 100$', r'$100 \leq$ RUL $\leq 200$',
                                                              r'RUL $> 200$']
    total_space_for_bars = 0.7
    bar_width = total_space_for_bars / len(levels)
    color_lst = ['skyblue', 'salmon'] if len(levels) == 2 else ['#72B7A1', '#CC79A7', '#E99675']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, group in enumerate(levels):
        color = color_lst[i % len(color_lst)]
        position = x + (i - len(levels) / 2 + 0.5) * bar_width
        ax.bar(position,
               sorted_mean_df.loc[group],
               bar_width,
               color=color,
               yerr=sorted_std_df.loc[group],
               label=group,
               capsize=5)

    ax.set_xlabel('Sensor', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(sensors, fontsize=18, rotation=45)
    ax.tick_params(axis='y', labelsize=18)  # Set font size for y-axis ticks
    ax.legend(fontsize=18)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join(save_path, f'sensor_freq_{group_name}.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'sensor_freq_{group_name}.pdf'), bbox_inches='tight')
    plt.show()


def result_combine(ds_name):
    file_name = f'25-01-04-{ds_name}'
    dfs = []
    for seed in range(10):
        save_dir = f'../result/{file_name}/seed-{seed}/full_historical-sel_method-linear_approx-sel_ratio-0.2/train-False'
        df = pd.read_csv(os.path.join(save_dir, 'result.csv'))
        df = _result_preprocess(df, ds_name=ds_name)
        df['seed'] = seed
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df


if __name__ == "__main__":
    for ds_name in ['FD001', 'FD003']:
        combined_df = result_combine(ds_name)
        plot_sensor_frequencies_no_group(ds_name, combined_df, seed_column='seed')
        if ds_name == 'FD003':
            plot_sensor_frequencies_groups(combined_df, group_name='fm', seed_column='seed')
            plot_sensor_frequencies_groups(combined_df, group_name='rul_range', seed_column='seed')
