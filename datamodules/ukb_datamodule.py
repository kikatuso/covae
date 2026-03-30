import lightning as L
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

from utils.dataset import UKB_dataset
from utils.utils import ResumableDataLoader


class UKBDataModule(L.LightningDataModule):
    def __init__(self, batch_size, num_workers=0,train_size=0.8,seed=42, data_dir: str = "/gpfs3/well/papiez/users/zwk579/.temp_data/256x256px/"):
        super().__init__()
        self.batch_size = batch_size
        self.train_size = train_size
        self.seed = seed
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    def setup(self, stage: str):
        dataset = UKB_dataset(self.data_dir, transform=self.transform)
        train_size = int(self.train_size * len(dataset))
        test_size = len(dataset) - train_size
        splits = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(self.seed))
        if stage == "fit":
            self.train = splits[0]
            self.fid = splits[0]
            self.test = splits[1]
        if stage == "test":
            self.test = splits[1]
        if stage == "predict":
            self.predict = splits[1]

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

if __name__ == "__main__":
    dm = UKBDataModule(batch_size=32)
    dm.setup(stage="fit")
    for batch in dm.train_dataloader():
        print(batch.shape)
        break