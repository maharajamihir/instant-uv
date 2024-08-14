import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class InstantUVDataset(Dataset):
    """
    Dataset class for handling UV, RGB, and optional 3D point data.

    Args:
        uv (tensor): Tensor containing UV coordinates.
        rgb (tensor): Tensor containing RGB values.
        points_xyz (tensor, optional): Tensor containing 3D point coordinates. Default is None.
    """

    def __init__(self, uv, rgb, points_xyz=None, angles=None, angles2=None, coords3d=None):
        # points_xyz are optional because we don't really need them for training.
        self.uv = torch.from_numpy(uv)
        self.rgb = torch.from_numpy(rgb)
        self.points_xyz = torch.from_numpy(points_xyz) if points_xyz is not None else None
        self.angles = torch.from_numpy(angles) if angles is not None else None
        self.angles2 = torch.from_numpy(angles2) if angles2 is not None else None
        self.coords3d = torch.from_numpy(coords3d) if coords3d is not None else None

    def __len__(self):
        """
        Returns the length of the dataset.
        
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.uv)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            dict: A dictionary containing the UV coordinates, RGB values, 
                  and optionally the 3D point coordinates for the sample.
        """
        sample = {
            'uv': self.uv[idx],
            'rgb': self.rgb[idx],
        }
        if self.points_xyz is not None:
            sample['xyz'] = self.points_xyz[idx]
        if self.angles is not None:
            sample['angles'] = self.angles[idx]
        if self.angles2 is not None:
            sample['angles2'] = self.angles2[idx]
        if self.coords3d is not None:
            sample['coords3d'] = self.coords3d[idx]
        return sample


class InstantUVDataLoader(DataLoader):
    """
    DataLoader class for batching and shuffling dataset samples.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Default is True.
    """

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.n_samples = len(dataset)
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size

        self.i = 0
        self.idxs = torch.arange(self.n_samples)

    def __len__(self):
        """
        Returns the number of batches per epoch.
        
        Returns:
            int: Number of batches per epoch.
        """
        return self.n_batches

    def __iter__(self):
        """
        Returns an iterator over the dataset.

        If shuffle is True, the dataset indices are shuffled at the beginning of each epoch.
        
        Returns:
            InstantUVDataLoader: The DataLoader instance.
        """
        if self.shuffle:
            self.idxs = torch.randperm(self.n_samples)
        self.i = 0
        return self

    def _get_next_batch_idxs(self):
        """
        Gets the indices for the next batch of samples.
        
        Returns:
            tensor: A tensor of indices for the next batch of samples.
        """
        low = self.i * self.batch_size
        high = min((self.i + 1) * self.batch_size, self.n_samples)
        self.i += 1
        return self.idxs[low:high]

    def __next__(self):
        """
        Retrieves the next batch of samples from the dataset.

        Returns:
            dict: A dictionary containing batches of UV coordinates, RGB values, 
                  and optionally 3D point coordinates.
        
        Raises:
            StopIteration: If there are no more batches to return.
        """
        if self.i >= self.n_batches:
            raise StopIteration

        batch_idxs = self._get_next_batch_idxs()
        batch = self.dataset.__getitem__(batch_idxs)

        return batch


# EXPERIMENTAL

class WeightedDataLoader(DataLoader):
    """
    DataLoader class for batching and shuffling dataset samples with importance sampling based on density.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (int): Number of samples per batch.
        shuffle (bool, optional): Whether to shuffle the data at every epoch. Default is True.
        grid_size (int, optional): Number of grid cells per dimension for density estimation. Default is 10.
        use_weights (bool, optional): Whether to use importance sampling with weights. Default is True.
    """

    def __init__(self, dataset, batch_size, grid_size=20):
        self.grid_size = grid_size
        self.weights = self._calculate_weights(dataset)
        sampler = WeightedRandomSampler(self.weights, len(self.weights), replacement=True)
        super().__init__(dataset, batch_size=batch_size, sampler=sampler)

        self.n_samples = len(dataset)
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size

    def _calculate_weights(self, dataset):
        """
        Calculate sampling weights based on the density of points in grid cells.
        """
        uv = dataset.uv.numpy()  # Convert tensor to numpy array

        # Normalize coordinates to grid size
        grid_x = (uv[:, 0] * self.grid_size).astype(int)
        grid_y = (uv[:, 1] * self.grid_size).astype(int)

        # Calculate grid cell counts
        grid_counts = np.zeros((self.grid_size, self.grid_size), dtype=int)
        np.add.at(grid_counts, (grid_x, grid_y), 1)

        # Calculate weights
        weights = 1.0 / grid_counts[grid_x, grid_y]

        # Normalize weights
        weights = weights.astype(np.float32)
        weights /= weights.sum()

        return weights



""" EXPERIMENTAL """


class InstantUVDataset2(Dataset):
    """
    Dataset class for handling UV, RGB, and optional 3D point data.

    Args:
        uv (tensor): Tensor containing UV coordinates.
        rgb (tensor): Tensor containing RGB values.
        points_xyz (tensor, optional): Tensor containing 3D point coordinates. Default is None.
    """

    def __init__(self, uv, rgb):
        # points_xyz are optional because we don't really need them for training.
        self.uv = np.array([torch.from_numpy(u) for u in uv], dtype=object)
        self.rgb = torch.from_numpy(rgb)

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.uv)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the UV coordinates, RGB values,
                  and optionally the 3D point coordinates for the sample.
        """
        sample = {
            'uv': self.uv[idx],
            'rgb': self.rgb[idx],
        }
        return sample