import os
import sys
import time
import glob
import math
import random
import numpy as np

# add script's dir to path to enable importing files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import render_functions as rf


objs = glob.glob("data/ShapeNetCore.v2/**/*.obj", recursive=True)
imgs = glob.glob("data/vot/**/*.jpg", recursive=True)
texs = glob.glob("data/textures/textures_train/*.jpg")
output = "data/generated/random"

n_images = 20
resolution = 320, 240
n_frames = 24

z_range = -8, -3
max_rot = math.pi / 6


rf.init(n_frames, resolution)
fr_tan, fr_offset = rf.calc_frustum()

st = time.time()

for i in range(n_images):
    rot_start = np.random.rand(3) * 2 * np.pi
    rf.render(
        os.path.join(output, f"{i:04}.webp"),
        random.choice(objs),
        random.choice(imgs),
        random.choice(texs),
        rf.gen_frustum_point(z_range, resolution, fr_tan, fr_offset),
        rf.gen_frustum_point(z_range, resolution, fr_tan, fr_offset),
        rot_start,
        rot_start + max_rot * (np.random.rand(3) * 2 - 1),
    )

st = time.time() - st
print(f"Rendered {n_images} images in {int(st)} seconds, {st / n_images:.2f}s per image")
