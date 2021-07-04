from pytorch_lightning import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

import torch


def _log_video_batch(trainer, name, video_batch, fps=24):
    vids = torch.cat(tuple(iter(video_batch)), -2)
    rgb, alpha = vids[:, :3], vids[:, 3:]
    vids = alpha * rgb + (1 - alpha)
    trainer.logger.experiment.add_video(
        name, vids[None], fps=fps, global_step=trainer.current_epoch
    )


def _load_on_device(dataloader, device):
    for batch in dataloader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        yield batch


class LogGTvsRenders(Callback):
    @rank_zero_only
    def on_validation_end(self, trainer, module):
        data = next(_load_on_device(module.val_dataloader(), module.device))
        gt = data["frames"]
        renders = module(data)["renders"]
        vids = torch.cat((gt, renders), -1)[:5]
        _log_video_batch(trainer, "gt_vs_renders", vids)


class LogPrediction(Callback):
    @rank_zero_only
    def on_validation_end(self, trainer, module):
        dl = module.predict_dataloader()
        if dl:
            data = next(_load_on_device(dl[0], module.device))
            renders = module(data)["renders"]
            _log_video_batch(trainer, "predictions", renders)
