import pytorch_lightning as pl

from torch.utils.data import DataLoader


class FMOData(pl.LightningDataModule):
    def __init__(self, train_data, valid_data, test_data=None):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=5, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=32, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=32, num_workers=1)
