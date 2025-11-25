# DSS4RemotePrognostic

This repository contains the Python code of our paper titled "Dynamic Sensor Selection for Remote Prognostics".

## Files

```{}
dataset/                               # Collected datasets
    AircraftEngine/CMAPSS/             # CMAPSS jet engine simulation data
src/                                   # Python source code
    data/                              # Data preprocessing scripts
        JetEngine.py                   # CMAPSS data preparation module
        util.py                        # Utility functions for preprocessing
    dataloader/                        # Data loading utilities
        sequence_dataloader.py         # Sequence data loader
    models/                            
        lstm.py                        # LSTM model 
    config.py                          # Configuration settings
    local_linear_approximator.py       # Local linear approximation module
    subset_closest_sum_solver.py       # Sensor selection solver
    train_evaluate.py                  # Model training and evaluation functions
    sensor_selection.py                # Sensor selection procedure
    exp.py                             # Experiment runner
    result_analysis.py                 # Final result analysis
    result_analysis_plot.py            # Visualizations and result plotting
result/                                # Result 
```



## Requirements

```{}
python = 3.9.12
pandas = 1.5.2
numpy = 1.21.5
matplotlib = 3.5.3
scipy = 1.7.3
scikit-learn = 1.3.2
seaborn = 0.11.2
torch = 1.12.0
tqdm = 4.67.1

```


## Usages

1. Run `src/sensor_selection.py` to obtain the dynamic sensor selection results.
2. Run `src/baseline_exp.py` to obtain the baseline results (static sensor selection and full sensor transmission).
3. Run `src/result_analysis.py` and `src/result_analysis_plot.py` to obtain the figures presented in the paper.

## Reference

If you find the code useful, please cite our paper:

```{}
@article{fu2025sensorselection,
  title={Dynamic Sensor Selection for Remote Prognostics},
  author={Fu, Ying and Huh, Ye Kwon and Liu, Kaibo},
  journal={IISE Transactions-just accepted},
  year={2025}}
```

