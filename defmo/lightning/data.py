import pytorch_lightning as pl

from torch.utils.data import DataLoader, Dataset


class FMOData(pl.LightningDataModule):
    def __init__(self, train_data, valid_data, predict_data=None, num_workers=1):
        super().__init__()
        self.train_data = train_data
        self.valid_data = valid_data
        self.predict_data = predict_data or []
        if isinstance(self.predict_data, Dataset):
            self.predict_data = [self.predict_data]
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=3, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=24, num_workers=self.num_workers)

    def predict_dataloader(self):
        return [
            DataLoader(pd, batch_size=24, num_workers=self.num_workers)
            for pd in self.predict_data
        ]
