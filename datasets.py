from defmo.lightning import FMOData

from defmo.training.data import BasicDataset, BackgroundAdder
from defmo.utils import FmoLoader, ZipLoader


def get_dataset(name, **kwargs):
    def double_blur():
        return FMOData(
            train_data=BasicDataset(
                FmoLoader("data/fmo_3_24_v1.zip", item_range=(0, 0.9), blurs=[1, 2]),
                BackgroundAdder(
                    ZipLoader("data/vot2018.zip", balance_subdirs=True),
                ),
            ),
            valid_data=BasicDataset(
                FmoLoader("data/fmo_3_24_v1.zip", item_range=(0.9, 1), blurs=[1, 2]),
                BackgroundAdder(
                    ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
                ),
            ),
        )

    try:
        return locals()[name](**kwargs)
    except KeyError:
        raise ValueError(f"Dataset '{name}' not found.")
