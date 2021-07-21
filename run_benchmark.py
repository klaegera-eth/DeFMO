import argparse

from defmo.benchmark import get_method, evaluate_on, helpers
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


method = get_method(
    "double_blur",
    DeFMO.load_from_checkpoint(args.checkpoint).to(args.device),
)

evaluate_on(
    helpers.get_falling_dataset("data/benchmark/falling_objects"),
    method,
    args,
    callback=lambda *args: hasattr(method, "callback") and method.callback(*args),
)
