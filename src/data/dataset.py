import torch
from torch.utils.data import Dataset, DataLoader

class InstantUVDataset(Dataset):
    """
    Dataset class for handling UV, RGB, and optional 3D point data.

    Args:
        uv (tensor): Tensor containing UV coordinates.
        rgb (tensor): Tensor containing RGB values.
        points_xyz (tensor, optional): Tensor containing 3D point coordinates. Default is None.
    """
    def __init__(self, uv, rgb, points_xyz=None):
        # points_xyz are optional because we don't really need them for training.
        self.uv = uv
        self.rgb = rgb
        self.points_xyz = points_xyz

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
