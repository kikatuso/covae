import lightning as L
from torch.utils.data import DataLoader, Dataset
import torch
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import CIFAR10

from utils.utils import rescaling, ResumableDataLoader

class LatentDataset(Dataset):
    def __init__(self, path):
        obj = torch.load(path, map_location="cpu")
        self.mu = obj["mu"]
        self.std = obj["std"]
        self.sample_std = obj["scale"]
    def __len__(self):
        return self.mu.shape[0]
    def __getitem__(self, i):
        return torch.distributions.Normal(self.mu[i], self.std[i]).sample([1])/self.sample_std


class LatentDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers=0, data_dir: str = "./", latent_data="./"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.latent_data = latent_data

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            self.train = LatentDataset(self.latent_data)
            self.fid = CIFAR10(self.data_dir, train=True, transform=transforms.ToTensor())

    def train_dataloader(self, shuffle=True):
        return ResumableDataLoader(self.train, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def fid_dataloader(self):
        return DataLoader(self.fid, batch_size=500, shuffle=False, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.predict, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
