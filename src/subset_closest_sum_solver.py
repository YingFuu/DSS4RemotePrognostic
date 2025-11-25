# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:59:26 2024

@author: Ying Fu
"""

import numpy as np
import math
import torch
import time


class DPsolver:
    def __init__(self, k, target, nums, start, cap=100000):
        self.k = k
        self.target = target
        self.nums = nums
        self.start = start
        self.cap = cap
        self.memo = {}

    def subset_sum_dp(self, k, target, nums, start):
        """
        Tensor version to find the subarray sum closest to the target.
        
        Parameters:
        k (int): Number of members to select from nums[start:] so that the sum 
                 of the selected members is closest to the target.
        target (float): The target sum.
        nums (torch.Tensor): All input numbers.
        start (int): Starting index.
        
        Returns:
        float: Gap to the target.
        torch.Tensor: Indices of selected members.
        """

        # No need to take or take all of it
        if k < 0 or k + start > len(nums):
            return math.inf, torch.tensor([])

        if k == 0:
            return target, torch.tensor([])

        if k + start == len(nums):
            return target - sum(nums[start:]), torch.tensor([i for i in range(start, len(nums))])

        # Check if the result is already computed
        if (k, target, start) in self.memo:
            return self.memo[(k, target, start)]

        # Compute and save
        # Case 1: Take the current element
        take_gap, take_indices = self.subset_sum_dp(k - 1, target - nums[start], nums, start + 1)
        take_indices = torch.cat((torch.tensor([start]), take_indices))

        # Case 2: Do not take the current element
        no_take_gap, no_take_indices = self.subset_sum_dp(k, target, nums, start + 1)

        # Choose the option with the smaller gap
        if abs(take_gap) <= abs(no_take_gap):
            result = take_gap, take_indices
        else:
            result = no_take_gap, no_take_indices

        self.memo[(k, target, start)] = result

        return result

    def subset_sum_dp_discrete(self):
        """
        Discretize the problem and find the subarray sum closest to the target.
        
        Returns:
        tuple: (gap to the target, indices of selected members)
        """

        if self.k <= 0:
            raise RuntimeError("k must be greater than 0")

        lb = sum(self.nums[self.nums < 0])
        ub = sum(self.nums[self.nums > 0])
        unit = max(abs(lb), ub) / self.cap

        nums_int = torch.tensor([int(i / unit) for i in self.nums], dtype=torch.int32)

        return self.subset_sum_dp(self.k, self.target, nums_int, self.start)


if __name__ == "__main__":
    d = torch.tensor([-2, 3, 5.5, -1, -0.3])
    goal = 0
    k = 3
    start = 0

    start_t = time.process_time()
    solver = DPsolver(k=k, target=goal, nums=d, start=start)
    res = solver.subset_sum_dp_discrete()
    print(res)
    print(f'Time taken: {time.process_time() - start_t:.6f} seconds')
