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


objs = rf.ZipLoader("data/ShapeNetCore.v2.zip", "*.obj", balance_subdirs=True)
texs = rf.ZipLoader("data/textures.zip", "*/textures_train/*.jpg")

output = "data/generated"

n_sequences = 5
resolution = 320, 240

n_frames = 24
blurs = [(0, 10), (-11, -1)]

z_range = -8, -3
delta_z = 1
delta_xy = 1, 3
max_rot = math.pi / 8


rf.init(n_frames, resolution)
frustum = rf.Frustum(z_range, resolution)
os.makedirs(output, exist_ok=True)

st = time.time()
with zipfile.ZipFile(os.path.join(output, str(uuid.uuid4()) + ".zip"), "w") as zip:
    for i in range(n_sequences):
        rot_start = np.random.rand(3) * 2 * np.pi
        with io.BytesIO() as outbuf, objs.get_random() as obj, texs.get_random() as tex:
            rf.render(
                outbuf,
                obj,
                tex,
                *frustum.gen_point_pair(delta_z, delta_xy),
                rot_start,
                rot_start + max_rot * (np.random.rand(3) * 2 - 1),
                blurs,
            )
            zip.writestr(f"{i:04}.webp", outbuf.getvalue())

st = time.time() - st
print(
    f"Rendered {n_sequences} sequences in {int(st)} seconds, {st / n_sequences:.2f}s per sequence"
)
