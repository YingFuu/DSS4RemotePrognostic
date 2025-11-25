# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:36:17 2024

@author: Ying Fu
"""
import pandas as pd
import numpy as np
import os

from matplotlib import pyplot as plt, tri

from util import rmse_score, append_row


def result_combine(result_dir, ds_name, seed_lst):
    """
    Combine summary CSV files from different seeds into a single summary CSV file.

    Parameters:
    result_dir (str): The directory where the result folders are located.
    ds_name (str): The dataset name used as a part of the directory names.
    seed_lst (list): A list of seed values used to create different result folders.
    """


def result_bs(ds_name, full):
    if full:
        ds_dir = f'../result/25-01-04-{ds_name}-full'
    else:
        ds_dir = f'../result/25-01-24-{ds_name}-static'
    rmse_lst = []
    for seed in range(10):
        file_dir = os.path.join(ds_dir, f'seed-{seed}/LSTM')
        df = pd.read_csv(os.path.join(file_dir, 'result-LSTM-test.csv'))
        rmse = rmse_score(df['pred_rul'], df['true_rul'])
        rmse_lst.append(rmse)
    return np.mean(rmse_lst), np.std(rmse_lst)


def result_dynamic(ds_name):
    def combine_result(ds_dir=f'../result/25-01-04-{ds_name}'):
        output_file = os.path.join(ds_dir, 'summary.csv')
        try:
            df_summary = pd.read_csv(output_file)
            print(f"Load summary file from {output_file}")
        except:
            df_lst = []
            for seed in range(1, 10):
                save_dir = os.path.join(ds_dir, f'seed-{seed}')
                summary_file = os.path.join(save_dir, 'summary.csv')
                df = pd.read_csv(summary_file)
                df_lst.append(df)

            df_summary = pd.concat(df_lst, ignore_index=True)
            output_file = os.path.join(ds_dir, 'summary.csv')
            df_summary.to_csv(output_file, index=False)
            print(f"Summary file saved to {output_file}")

        return df_summary

    df1 = combine_result(ds_dir=f'../result/25-01-04-{ds_name}')
    df2 = combine_result(ds_dir=f'../result/25-05-11-{ds_name}')
    df_summary = pd.concat([df1, df2], ignore_index=True)

    return df_summary


def plot_rmse_ratio(ds_name):
    mean, std = result_bs(ds_name, full=True)
    df_summary = result_dynamic(ds_name)
    sel_ratio_lst = [0.1, 0.2, 0.4, 0.8]
    method_lst = ['linear_approx', 'random', 'greedy', 'genetic']
    initial_units_nums_ratio_lst = [1]
    df_summary_filtered = df_summary[
        df_summary['initial_units_nums_ratio'].isin(initial_units_nums_ratio_lst) &
        df_summary['sel_ratio'].isin(sel_ratio_lst) &
        df_summary['method'].isin(method_lst)
        ].reset_index(drop=True)

    method_line_dict = {
        'linear_approx': 'solid',
        'random': 'dashed',
        'greedy': 'dashdot',
        'genetic': 'dotted'
    }
    method_color_dict = {
        'linear_approx': 'tab:red',
        'random': 'tab:orange',
        'greedy': 'tab:green',
        'genetic': 'tab:blue',
    }

    plt.figure(figsize=(9, 6))
    for method in df_summary_filtered['method'].unique():
        sub = df_summary_filtered[df_summary_filtered['method'] == method].reset_index(drop=True)
        sub_stat = sub.groupby('sel_ratio')['rmse_test'].agg(['mean', 'std']).reset_index()

        new_row = pd.Series({'sel_ratio': 1,
                             'mean': mean,
                             'std': std,
                             })
        sub_stat = append_row(sub_stat, new_row)

        plt.plot(sub_stat['sel_ratio'], sub_stat['mean'],
                 linestyle=method_line_dict[method],
                 color=method_color_dict[method],
                 linewidth=4, label=method, marker='o', markersize=8)
        plt.fill_between(sub_stat['sel_ratio'],
                         sub_stat['mean'] - sub_stat['std'],
                         sub_stat['mean'] + sub_stat['std'],
                         color=method_color_dict[method],
                         alpha=0.2)
        plt.grid(True, which='major', linestyle='--', alpha=0.4)
        plt.locator_params(axis='y', nbins=7)
        plt.locator_params(axis='x', nbins=5)
        plt.xlim(0, 1.1)
        plt.xlabel('transmission ratio')
        plt.ylabel('RMSE')

    if ds_name in ['FD001', 'FD003']:
        x = 11 / 15
    else:
        x = 11 / 21
    y, yerr = result_bs(ds_name, full=False)

    plt.errorbar(x, y, yerr=yerr, fmt='h', markersize=8, capsize=8,
                 elinewidth=3, capthick=3, label='static', color='tab:purple')
    plt.legend(loc='best')
    save_path = '../figures'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{ds_name}_sel_ratio.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'{ds_name}_sel_ratio.pdf'), bbox_inches='tight')
    plt.show()


def contour_plots(ds_name='FD001'):
    df_summary = result_dynamic(ds_name)
    method_lst = ['linear_approx']
    sel_ratio_lst = [0.1]
    initial_units_nums_ratio_lst = [0, 0.2, 0.4, 0.8, 1]
    update_period_ratio_lst = [0, 0.1, 0.2, 0.4, 0.8]
    df_summary_filtered = df_summary[
        df_summary['initial_units_nums_ratio'].isin(initial_units_nums_ratio_lst) &
        df_summary['update_period_ratio'].isin(update_period_ratio_lst) &
        df_summary['sel_ratio'].isin(sel_ratio_lst) &
        df_summary['method'].isin(method_lst)
        ].reset_index(drop=True)

    group_result = df_summary_filtered.groupby(['initial_units_nums', 'update_period'])['rmse_test'].agg(
        ['mean', 'std']).reset_index()

    plt.figure(figsize=(12, 9))
    # Create a grid for initial_units and update_period
    xi = np.linspace(group_result['initial_units_nums'].min(), group_result['initial_units_nums'].max(), 100)
    yi = np.linspace(group_result['update_period'].min(), group_result['update_period'].max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate rmse_mean over the grid
    triang = tri.Triangulation(group_result['initial_units_nums'], group_result['update_period'])
    interpolator = tri.LinearTriInterpolator(triang, group_result['mean'])
    zi = interpolator(xi, yi)

    if ds_name in ['FD001', 'FD002']:
        vmin = 33
        vmax = 50
        cmap = 'viridis'
    else:
        vmin = 60
        vmax = 75
        cmap = 'magma'

    contour = plt.contourf(xi, yi, zi, levels=12, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(contour, label='RMSE')

    plt.xlabel('Historical Run-to-Failure Units')
    plt.ylabel('Update Cycle')
    plt.title(ds_name)
    save_path = '../figures'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{ds_name}_initial_update_num.svg'), bbox_inches='tight')
    plt.savefig(os.path.join(save_path, f'{ds_name}_initial_update_num.pdf'), bbox_inches='tight')
    plt.show()


def LLA_table(ds_name):
    ########## TODO need test
    results = {50: {'rmse': [], 'time': []},
               100: {'rmse': [], 'time': []},
               200: {'rmse': [], 'time': []}}

    for num_samples in [50, 100, 200]:
        for noise_std in [0.0001, 0.01, 0.05, 0.4, 3]:
            rmse_lst = []
            time_lst = []

            for seed in range(10):
                save_dir = os.path.join(f'../result/25-01-20-{ds_name}-LLA/N-{num_samples}-std-{noise_std}',
                                        f'seed-{seed}')
                df = pd.read_csv(os.path.join(save_dir, 'summary.csv'))
                rmse_lst.append(df['rmse_test'][0])
                time_lst.append(df['time(s)'][0])

            # Calculate mean and std
            rmse_mean, rmse_std = np.mean(rmse_lst), np.std(rmse_lst)
            time_mean, time_std = np.mean(time_lst), np.std(time_lst)

            print(f'{num_samples = }, {noise_std = }, {rmse_mean = }, {time_mean = }')

            results[num_samples]['rmse'].append(f"{rmse_mean:.4f} ± {rmse_std:.4f}")
            results[num_samples]['time'].append(f"{time_mean:.2f} ± {time_std:.2f}")

    noise_levels = [0.0001, 0.01, 0.05, 0.4, 3]
    df_combined = pd.DataFrame({
        '50_rmse': results[50]['rmse'],
        '50_time': results[50]['time'],
        '100_rmse': results[100]['rmse'],
        '100_time': results[100]['time'],
        '200_rmse': results[200]['rmse'],
        '200_time': results[200]['time']
    }, index=noise_levels)

    print("\nTable 3: RMSE and computation time for different N and σ on FD3D64:\n")
    print("=" * 120)
    print(f"{'':12} |{'50':^35}|{'100':^35}|{'200':^35}")
    print("-" * 120)
    print(f"{'noise_std':12} |{'RMSE':^17}{'Time(s)':^18}|{'RMSE':^17}{'Time(s)':^18}|{'RMSE':^17}{'Time(s)':^18}")
    print("-" * 120)

    for idx in df_combined.index:
        row = df_combined.loc[idx]
        print(f"{idx:12} |{row['50_rmse']:^17}{row['50_time']:^18}|"
              f"{row['100_rmse']:^17}{row['100_time']:^18}|"
              f"{row['200_rmse']:^17}{row['200_time']:^18}")

    print("=" * 120)


def LLA_width(ds_name):
    for scale_factor in [0.01, 0.1, 1, 2, 10]:
        rmse_lst = []
        time_lst = []

        for seed in range(10):
            save_dir = f'../result/25-05-07-{ds_name}/scale_factor-{scale_factor}/seed-{seed}'
            df = pd.read_csv(os.path.join(save_dir, 'summary.csv'))
            rmse_lst.append(df['rmse_test'][0])
            time_lst.append(df['time(s)'][0])

        # Calculate mean and std
        rmse_mean, rmse_std = np.mean(rmse_lst), np.std(rmse_lst)
        # time_mean, time_std = np.mean(time_lst), np.std(time_lst)

        print(f'{scale_factor = }, {rmse_mean = }, {rmse_std = }')


def format_results_with_stats(mean_df, std_df, decimal_places=3):
    """
    Format the results to show mean ± std for each sensor and fm combination.
    
    Parameters:
    mean_df (pandas.DataFrame): DataFrame containing mean values
    std_df (pandas.DataFrame): DataFrame containing standard deviation values
    decimal_places (int): Number of decimal places to round to
    
    Returns:
    pandas.DataFrame: Formatted DataFrame with mean ± std values
    """
    formatted_df = pd.DataFrame(index=mean_df.index, columns=mean_df.columns)

    for col in mean_df.columns:
        formatted_df[col] = (mean_df[col].round(decimal_places).astype(str) +
                             ' ± ' +
                             std_df[col].round(decimal_places).astype(str))
    return formatted_df


if __name__ == "__main__":
    # LLA
    # LLA_width('FD003')
    # LLA_table('FD003')

    plot_rmse_ratio('FD001')
    plot_rmse_ratio('FD002')
    plot_rmse_ratio('FD003')
    plot_rmse_ratio('FD004')

    # contour plot
    contour_plots('FD001')
    contour_plots('FD002')
    contour_plots('FD003')
    contour_plots('FD004')
