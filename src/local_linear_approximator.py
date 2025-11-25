# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:56:30 2024

@author: Ying Fu
"""

import torch


class LinearApproximator:
    def __init__(self, num_samples=100, noise_std=0.05, scale_factor=2):

        """
        Parameters:
        num_samples (int): Number of samples to generate. Default is 100.
        noise_std (float): Standard deviation of the noise to be added to the samples. Default is 0.05.
        """
        self.num_samples = num_samples
        self.noise_std = noise_std
        self.scale_factor = scale_factor

    def generate_samples(self, historical_data, x, net):
        """
         Generate sample data for linear approximation.
         
         Parameters:
         historical_data (torch.Tensor): Historical data to be used in the sampling.
         x (torch.Tensor): Input vector for generating mean values.
         model (torch.nn.Module): The model to generate predictions.
         scale_factor (float): Scale factor for weighting the samples. Default is 2.
         
         Returns:
         torch.Tensor: Generated input.
         torch.Tensor: Generated target values.
         torch.Tensor: Weights for the samples.
         """

        mean = x.repeat(self.num_samples, 1)  # Repeat num_samples times
        X = torch.normal(mean=mean, std=self.noise_std)  # Generate data used for linear approximation
        distances = torch.sqrt(torch.sum((X - mean) ** 2, axis=1))
        weights = torch.exp(-distances / self.scale_factor)

        y = torch.zeros(self.num_samples, 1)  # Generated target used for linear approximation
        for i in range(self.num_samples):
            sample_instance = torch.cat((historical_data, X[i].unsqueeze(0)), dim=0)
            mean_prediction = net.get_prediction_one(sample_instance)
            y[i, 0] = mean_prediction

        return X, y, weights

    @staticmethod
    def perform_linear_regression(X, y, weights):
        """
        Perform Linear Regression using the given samples and weights.

        LR in Pytorch:
            1. Direct Solution: Best for small datasets
            2. Gradient Descent: Better for large datasets or for adding complexity like regularization
            3. LBFGS: Ideal for medium-sized problems where direct methods might struggle with memory issues,
                      but the dataset is still small enough to handle efficiently within the optimizer.

        Parameters:
        X (torch.Tensor): Input features, shape (n_samples, n_features).
        y (torch.Tensor): Target values, shape (n_samples, 1).
        weights (torch.Tensor): Sample weights, shape (n_samples,).

        Returns:
        torch.Tensor: Coefficients (including bias as the first element).
        """
        ones = torch.ones(X.shape[0], 1, device=X.device)
        X_bias = torch.cat([ones, X], dim=1)
        W = torch.diag(weights)
        XTWX = X_bias.T @ W @ X_bias
        XTWy = X_bias.T @ W @ y

        beta = torch.linalg.solve(XTWX, XTWy)  # contains both the intercept and slopes

        beta = beta.squeeze(1)  # --->[num_features+1]
        # intercept = beta[0]
        # slopes = beta[1:]
        return beta

    @staticmethod
    def evaluate_regression(y, y_pred):
        """
        Calculate MAPE and R-square to evaluate the performance of Linear Regression.
        
        Args:
            y (torch.Tensor): Ground truth target values.
            y_pred (torch.Tensor): Predicted target values.
        
        Returns:
            mape (float): Mean Absolute Percentage Error.
            r2 (float): R-squared value.
        """
        mape = torch.mean(torch.abs((y - y_pred) / y)) * 100

        # R-squared
        ss_total = torch.sum((y - torch.mean(y)) ** 2)
        ss_residual = torch.sum((y - y_pred) ** 2)

        if ss_total == 0:
            # All true values are the same
            r2 = 1.0 if ss_residual == 0 else 0.0
        else:
            r2 = 1 - ss_residual / ss_total

        return mape, r2
