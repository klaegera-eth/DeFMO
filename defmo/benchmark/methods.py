import torch
import numpy as np
from torchvision.transforms.functional import to_tensor, to_pil_image

from .submodule.benchmark import loaders_helpers as helpers
from .submodule.benchmark.benchmark_loader import evaluate_on


def benchmark(model, data, method, args, method_kwargs={}):
    method = get_method(method, model, **method_kwargs)
    callback = lambda *args: hasattr(method, "callback") and method.callback(*args)
    return evaluate_on(data, method, args, callback)


def get_method(name, model, shape=(320, 240), sub_steps=5):
    def double_blur(img, bg, bbox, n_frames, radius, _):
        if not hasattr(double_blur, "state"):
            double_blur.state = {}
            double_blur.callback = lambda *_: double_blur.state.clear()
        state = double_blur.state

        if "prev_img" not in state:
            state["prev_img"] = img
            state["prev_bbox"] = bbox
            return None, None

        bbox_union = np.minimum(state["prev_bbox"], bbox)
        bbox_union[2:] = np.maximum(state["prev_bbox"], bbox)[2:]
        bbox_extended = helpers.extend_bbox(
            bbox_union, 4 * radius, shape[1] / shape[0], img.shape
        )

        crop = lambda x: helpers.crop_resize(x, bbox_extended, shape)
        rev_crop = lambda x: helpers.rev_crop_resize(x, bbox_extended, img)

        inputs = to_tensor(crop(state["prev_img"])), to_tensor(crop(img))

        state["prev_img"] = img
        state["prev_bbox"] = bbox

        with torch.no_grad():
            was_training = model.training
            model.eval()
            batch = torch.stack(inputs)[None].float().to(model.device)
            renders = model(batch, n_frames * 2 * sub_steps)["renders"]
            model.train(was_training)

        renders = (
            renders[0, -n_frames * sub_steps :]
            .reshape(n_frames, sub_steps, *renders.shape[2:])
            .mean(1)
            .cpu()
            .numpy()
            .transpose((2, 3, 1, 0))
        )

        return rev_crop(helpers.rgba2hs(renders, crop(bg))), None

    def dummy_img(img, bg, bbox, n_frames, *_):
        return np.repeat(img[:, :, :, None], n_frames, 3), None

    def dummy_bg(img, bg, bbox, n_frames, *_):
        return np.repeat(bg[:, :, :, None], n_frames, 3), None

    try:
        return locals()[name]
    except KeyError:
        raise ValueError(f"Benchmark method '{name}' not found.")


def show(*imgs):
    for img in imgs:
        if not isinstance(img, torch.Tensor):
            img = to_tensor(img)
        to_pil_image(img).show()
    print("showing, enter to continue")
    input()
