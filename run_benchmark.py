import argparse
import sys
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image

sys.path.append("benchmark_module")

from benchmark.benchmark_loader import evaluate_on
import benchmark.loaders_helpers as lh

from defmo.training import DeFMO


parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--device", default="cuda")
parser.add_argument("--verbose", default=False)
parser.add_argument("--save_visualization", default=False)
parser.add_argument("--visualization_path", default="benchmark_viz")
parser.add_argument("--add_traj", default=False)
parser.add_argument("--method_name", required=False, default="checkpoint")
args = parser.parse_args()


de = DeFMO.load_from_checkpoint(args.checkpoint).to(args.device)
de.eval()

shape = 320, 240
n_average = 5

prev = None


def show(*imgs):
    for img in imgs:
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)
        to_pil_image(img).show()
    print("showing, enter to continue")
    input()


def method(I, B, bbox_tight, n_frames, radius, _):
    global prev
    if prev is None:
        prev = I, bbox_tight
        return None, None

    prev_I, prev_bbox = prev

    bbox_combined = np.minimum(prev_bbox, bbox_tight)
    bbox_combined[2:] = np.maximum(prev_bbox, bbox_tight)[2:]

    # crop input
    bbox = lh.extend_bbox(bbox_combined, 4 * radius, shape[1] / shape[0], I.shape)
    I_crop = lh.crop_resize(I, bbox, shape)
    prev_I_crop = lh.crop_resize(prev_I, bbox, shape)
    B_crop = lh.crop_resize(B, bbox, shape)

    # show(
    #     lh.crop_resize(prev_I, prev_bbox, shape),
    #     lh.crop_resize(I, bbox_tight, shape),
    #     lh.crop_resize(prev_I, bbox_combined, shape),
    #     lh.crop_resize(I, bbox_combined, shape),
    #     prev_I_crop,
    #     I_crop,
    # )

    # predict
    batch = torch.stack((to_tensor(prev_I_crop), to_tensor(I_crop)), 0)[None].float()
    with torch.no_grad():
        renders = de(batch.to(de.device), n_frames * 2 * n_average)["renders"].cpu()
        renders = (
            renders[0, -n_frames * n_average :]
            .reshape(n_frames, n_average, *renders.shape[2:])
            .mean(1)
        )
        renders = renders.numpy().transpose((2, 3, 1, 0))

    # reverse crop
    HS_crop = lh.rgba2hs(renders, B_crop)
    HS = lh.rev_crop_resize(HS_crop, bbox, I)

    prev = I, bbox_tight

    return HS, None


def callback(*_):
    global prev
    prev = None


evaluate_on(
    lh.get_falling_dataset("data/benchmark/falling_objects"),
    method,
    args,
    callback,
)
