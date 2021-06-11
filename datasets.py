from defmo.lightning import FMOData

from defmo.training.data import BasicDataset, BackgroundAdder
from defmo.utils import FmoLoader, ZipLoader


def get_dataset(name, num_workers=1, **kwargs):
    def double_blur(train_range=(0, 0.9), valid_range=(0.9, 1)):
        return FMOData(
            train_data=BasicDataset(
                FmoLoader("data/fmo_3_24_v1.zip", item_range=train_range, blurs=[1, 2]),
                BackgroundAdder(
                    ZipLoader("data/vot2018.zip", balance_subdirs=True),
                ),
            ),
            valid_data=BasicDataset(
                FmoLoader("data/fmo_3_24_v1.zip", item_range=valid_range, blurs=[1, 2]),
                BackgroundAdder(
                    ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
                ),
            ),
            num_workers=num_workers,
        )

    try:
        if num_workers > 1:
            from torch.multiprocessing import set_start_method

            set_start_method("spawn", force=True)
        return locals()[name](**kwargs)
    except KeyError:
        raise ValueError(f"Dataset '{name}' not found.")
