from defmo.training.data import DataModule, MultiDataset, BackgroundAdder
from defmo.utils import FmoLoader, ZipLoader


def get_dataset(name, **kwargs):
    def double_blur(**kwargs):
        return _dataset(**{"blurs": [1, 2], **kwargs})

    def multi_object(**kwargs):
        return double_blur(**{"num_objects": [1, 2, 3], **kwargs})

    def test(**kwargs):
        return double_blur(**{"train_range": (0, 9), "valid_range": (9, 57), **kwargs})

    def single_blur(**kwargs):
        return _dataset(**{"blurs": [0], **kwargs})

    def test_single(**kwargs):
        return single_blur(**{"train_range": (0, 9), "valid_range": (9, 57), **kwargs})

    try:
        if kwargs.get("num_workers", 1) > 1:
            from torch.multiprocessing import set_start_method

            set_start_method("spawn", force=True)
        return locals()[name](**kwargs)
    except KeyError:
        raise ValueError(f"Dataset '{name}' not found.")


def _dataset(
    blurs,
    num_objects=1,
    train_range=(0, 0.9),
    valid_range=(0.9, 1),
    num_workers=1,
):
    return DataModule(
        train_data=MultiDataset(
            FmoLoader("data/fmo_3_24_v1.zip", item_range=train_range, blurs=blurs),
            BackgroundAdder(
                ZipLoader("data/vot2018.zip", balance_subdirs=True),
            ),
            num_objects=num_objects,
        ),
        valid_data=MultiDataset(
            FmoLoader("data/fmo_3_24_v1.zip", item_range=valid_range, blurs=blurs),
            BackgroundAdder(
                ZipLoader("data/otb.zip", filter="*.jpg", balance_subdirs=True),
            ),
            num_objects=num_objects,
        ),
        predict_data=MultiDataset(
            FmoLoader(
                "data/falling.zip", dummy_frames=24, blurs=[x - 1 for x in blurs]
            ),
            background_adder=None,
        ),
        num_workers=num_workers,
    )
