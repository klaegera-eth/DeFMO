import pytorch_lightning as pl

from torch.utils.data import DataLoader


class FMOData(pl.LightningDataModule):
    def __init__(self, train_data, valid_data):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=4, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=24, num_workers=1)
