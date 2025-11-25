# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:52:26 2023

@author: Ying Fu
"""
import torch
import pandas as pd
from sklearn.model_selection import KFold
from util import tensor2numpy


# Define a custom KFold iterator to split at the unit level
class kFold:
    def __init__(self, df, n_splits):
        self.n_splits = n_splits
        self.unit_ids = df['id'].unique()

    def split(self, df):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for train_unit_ids, test_unit_ids in kf.split(self.unit_ids):
            train_units = df[df['id'].isin(train_unit_ids)]
            test_units = df[df['id'].isin(test_unit_ids)]
            yield train_units, test_units


def train_step(train_loader, device, model, loss_function, optimizer):
    """
    train a model
    :param train_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: a specified model, LSTM or CNN, or any custom models.
    :param loss_function:
    :param optimizer:
    :return: float, average train loss
    """
    num_batches = len(train_loader)

    total_loss = 0
    model.train()
    for X, y, _ in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        output = model(X).to(device)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    return avg_loss


def val_step(val_loader, device, model, loss_function):
    """
    evaluate a model
    :param val_loader: DataLoader class
    :param device: run on cpu or CUDA
    :param model: already trained model
    :param loss_function:
    :return: float, average validation loss
    """
    num_batches = len(val_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():  # will not use CUDA memory
        for X, y, _ in val_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X).to(device)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches

    return avg_loss


def predict_step(test_loader, device, model):
    """
    predict the model
    :param test_loader: DataLoader class
    :param model: already trained model.
    :param device: run on cpu or CUDA
    :return: prediction and true, torch.tensor
    """
    num_batches = len(test_loader)
    batch_size = test_loader.batch_size
    num_elements = len(test_loader.dataset)
    result = torch.zeros((num_elements, 3))
    model.eval()
    with torch.no_grad():
        for i, (X, y, data_id) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            data_id = data_id.to(device)
            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            y_pred = model(X)

            result[start:end, 0] = data_id
            result[start:end, 1] = y_pred
            result[start:end, 2] = y

    result = tensor2numpy(result)
    result = pd.DataFrame(result, columns=['unit_id',
                                           'pred_rul',
                                           'true_rul'])
    return result



