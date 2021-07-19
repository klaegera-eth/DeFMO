import argparse
import sys
import torch
from torchvision.transforms.functional import to_tensor

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


def method(I, B, bbox, n_frames, radius, _):

    # crop input
    bbox = lh.extend_bbox(bbox.copy(), 4 * radius, shape[1] / shape[0], I.shape)
    I_crop = lh.crop_resize(I, bbox, shape)
    B_crop = lh.crop_resize(B, bbox, shape)

    # predict
    batch = torch.stack((to_tensor(I_crop), to_tensor(B_crop)), 0)[None].float()
    with torch.no_grad():
        renders = de(batch.to(de.device), n_frames)["renders"].cpu()
    renders = renders[0].numpy().transpose((2, 3, 1, 0))

    # reverse crop
    HS_crop = lh.rgba2hs(renders, B_crop)
    HS = lh.rev_crop_resize(HS_crop, bbox, I)

    return HS, None


evaluate_on(lh.get_falling_dataset("data/benchmark/falling_objects"), method, args)
