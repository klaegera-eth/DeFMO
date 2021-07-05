import os
from pytorch_lightning.callbacks import ModelCheckpoint


class ContinuousModelCheckpoint(ModelCheckpoint):
    """Extends ModelCheckpoint to properly reload internal best model
    tracking state when resuming training. This enables the best model
    checkpoint to be tracked and overwritten over multiple training sessions."""

    def __init__(self, path, monitor, save_last=True, **kwargs):
        dirpath, filename = os.path.split(os.path.realpath(path))
        filename, fileext = os.path.splitext(filename)
        super().__init__(
            dirpath,
            filename + "-best",
            monitor,
            save_last=save_last,
            **kwargs,
        )
        self.CHECKPOINT_NAME_LAST = filename
        self.FILE_EXTENSION = fileext

    def on_load_checkpoint(self, trainer, pl_module, callback_state) -> None:
        super().on_load_checkpoint(trainer, pl_module, callback_state)
        self.kth_best_model_path = self.best_model_path
        self.best_k_models[self.best_model_path] = self.best_model_score
