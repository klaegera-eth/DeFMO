import os
import sys
import time
import math
import numpy as np
import zipfile
import uuid
import io

# add script's dir to path to enable importing files
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import render_functions as rf


objs = rf.ZipLoader("data/ShapeNetCore.v2.zip", "*.obj")
imgs = rf.ZipLoader("data/vot2018.zip", "*.jpg")
texs = rf.ZipLoader("data/textures.zip", "*/textures_train/*.jpg")

output = "data/generated/random"

n_images = 20
resolution = 320, 240
n_frames = 24

z_range = -8, -3
max_rot = math.pi / 6


rf.init(n_frames, resolution)
fr_tan, fr_offset = rf.calc_frustum()

st = time.time()

with zipfile.ZipFile(os.path.join(output, str(uuid.uuid4()) + ".zip"), "w") as zip:
    for i in range(n_images):
        rot_start = np.random.rand(3) * 2 * np.pi
        with io.BytesIO() as outbuf, objs.get_random() as obj, imgs.get_random() as img, texs.get_random() as tex:
            rf.render(
                outbuf,
                obj,
                img,
                tex,
                rf.gen_frustum_point(z_range, resolution, fr_tan, fr_offset),
                rf.gen_frustum_point(z_range, resolution, fr_tan, fr_offset),
                rot_start,
                rot_start + max_rot * (np.random.rand(3) * 2 - 1),
            )
            zip.writestr(f"{i:04}.webp", outbuf.getvalue())

st = time.time() - st
print(f"Rendered {n_images} images in {int(st)} seconds, {st / n_images:.2f}s per image")
