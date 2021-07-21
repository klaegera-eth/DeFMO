from pytorch_lightning import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

import torch

from types import SimpleNamespace
from defmo.benchmark import benchmark


def _log_video_batch(trainer, name, video_batch, fps=24):
    vids = torch.cat(tuple(iter(video_batch)), -2)
    rgb, alpha = vids[:, :3], vids[:, 3:]
    vids = alpha * rgb + (1 - alpha)
    trainer.logger.experiment.add_video(
        name, vids[None], fps=fps, global_step=trainer.global_step
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


class LogBenchmark(Callback):
    def __init__(self, dataset, method, method_kwargs={}):
        self.dataset = dataset
        self.method = method
        self.method_kwargs = method_kwargs
        self.args = SimpleNamespace(
            verbose=True,
            save_visualization=False,
            visualization_path="",
            add_traj=False,
            method_name="benchmark_callback",
        )

    @rank_zero_only
    def on_validation_epoch_end(self, _, module):
        _, psnr, ssim = benchmark(
            module,
            self.dataset,
            self.method,
            self.args,
            self.method_kwargs,
        )
        module.log("benchmark/PSNR", psnr)
        module.log("benchmark/SSIM", ssim)
