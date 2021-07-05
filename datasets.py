from defmo.training.data import DataModule, MultiDataset, BackgroundAdder
from defmo.utils import FmoLoader, ZipLoader


def get_dataset(name, num_workers=1, **kwargs):
    def double_blur(**kwargs):
        return multi_object(n_obj=1, **kwargs)

    def multi_object(train_range=(0, 0.9), valid_range=(0.9, 1), n_obj=[1, 2, 3]):
        return DataModule(
            train_data=MultiDataset(
                FmoLoader("data/fmo_3_24_v1.zip", item_range=train_range, blurs=[1, 2]),
                BackgroundAdder(
                    ZipLoader("data/vot2018.zip", balance_subdirs=True),
                ),
                num_objects=n_obj,
            ),
            valid_data=MultiDataset(
                FmoLoader("data/fmo_3_24_v1.zip", item_range=valid_range, blurs=[1, 2]),
                BackgroundAdder(
                    ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
                ),
                num_objects=n_obj,
            ),
            predict_data=MultiDataset(
                FmoLoader("data/falling.zip", dummy_frames=24),
                background_adder=None,
            ),
            num_workers=num_workers,
        )

    def test():
        return double_blur(train_range=(0, 9), valid_range=(9, 57))

    try:
        if num_workers > 1:
            from torch.multiprocessing import set_start_method

            set_start_method("spawn", force=True)
        return locals()[name](**kwargs)
    except KeyError:
        raise ValueError(f"Dataset '{name}' not found.")
