import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """
    Sliding window for time series prediction
        sample: (window_size,features)
        label: float
        data_id: int, indicate this sample belongs to which engine
    """

    def __init__(self, df, ds_name, window_size, stride=1,
                 y_column=['RUL'], mean=None, std=None):

        """
        Initializes the dataset with normalization.
        
        Parameters:
        - df: DataFrame containing the data.
        - window_size: The size of each data window.
        - stride: The stride between windows.
        - no_train_col: Columns not to be used for training (e.g., non-feature columns).
        - y_column: The name of the label column.
        - mean: The mean used for normalization (optional).
        - std: The standard deviation used for normalization (optional).
        """

        df = df.reset_index(drop=True)

        if ds_name in ['FD001', 'FD003']:  # both of them have only one working condition (constant values)
            self.no_train_col = ['cycle', 'WC', 'Altitude', 'Mach_number', 'TRA']
        elif ds_name in ['FD002', 'FD004']:
            self.no_train_col = ['cycle']
        elif ds_name in ['Synthetic_FD003', 'Synthetic_FD001']:
            self.no_train_col = ['cycle', 'WC', 'failure_mode']
        else:
            raise RuntimeError(f'{ds_name} does not exist.')

        all_columns = list(df.columns)
        self.features = list(set(all_columns).difference(set(self.no_train_col + y_column + ['id'])))

        if 'WC' in self.features:
            self.features.remove('WC')
            self.features.append('WC')  # Ensure 'WC' is the last feature

        self.n_features = len(self.features)
        # print(f'Features: {self.features}, number of features: {self.n_features}')

        self.y_column = y_column
        self.window_size = window_size
        self.stride = stride
        self.dtype = torch.float

        # Normalize data
        self.df, self.mean, self.std = self._normalize(df, mean, std)
        self.data_tuples = self._create_data_tuples(self.df, self.window_size, self.stride, self.y_column)

    def _normalize(self, df, mean=None, std=None):
        """
        Normalizes the DataFrame except for the specified columns using given mean and std.
        If mean or std is None, they are computed from the DataFrame itself (useful for training data).
        """

        df_copy = df.copy()
        norm_cols = [col for col in self.features + self.y_column if col != 'WC']
        if mean is None or std is None:
            mean = df_copy[norm_cols].mean()
            std = df_copy[norm_cols].std()

        # print(f'{mean = }')
        # print(f'{std = }')
        df_copy[norm_cols] = (df_copy[norm_cols] - mean) / std

        return df_copy, mean, std

    def _create_data_tuples(self, df, window_size, stride, y_column):
        """
        Creates tuples of (sample, label, data_id) for each window in the dataset.
        """
        self.data_tuples = []
        X = df[self.features + ['id']]
        y = df[y_column]

        unique_ids = df['id'].unique()
        for data_id in unique_ids:
            X_sub = X[X['id'] == data_id]
            y_sub = y.iloc[X_sub.index]
            X_sub = X_sub.drop('id', axis=1).reset_index(drop=True)
            y_sub = y_sub.reset_index(drop=True)

            idxs = [i for i in range(0, len(X_sub) + 1 - self.window_size, self.stride)]
            for j in idxs:
                sample_df = X_sub.iloc[j: j + self.window_size, :]
                sample_copy = sample_df.copy().reset_index(drop=True)
                label_df = y_sub.iloc[j + self.window_size - 1: j + self.window_size]
                label_copy = label_df.copy().reset_index(drop=True)

                sample = torch.tensor(sample_copy.values, dtype=self.dtype)
                label = torch.tensor(label_copy.values, dtype=self.dtype)
                label = label.squeeze()
                data_id_t = torch.tensor(data_id)

                data_tuple = (sample, label, data_id_t)
                self.data_tuples.append(data_tuple)

        return self.data_tuples

    def __len__(self):
        return len(self.data_tuples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample, label, data_id = self.data_tuples[idx]

        return sample, label, data_id

    def add_new_sample(self, sample, label, data_id):
        """
        Add new sample directly to the dataset.
        """
        self.data_tuples.append((sample, label, data_id))

    def add_new_samples(self, new_data_tuples):
        """
        Add a set of new data tuples directly to the dataset.
        Each element in new_data_tuples should be a tuple of (sample, label, data_id).
        """
        for sample, label, data_id in new_data_tuples:
            self.add_new_sample(sample, label, data_id)

    def get_subset_by_data_id(self, data_id):
        """
        Filters the dataset to only include data tuples with the specified data_id.
        """
        return [t for t in self.data_tuples if t[2] == data_id]
