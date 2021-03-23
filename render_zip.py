import os
import sys
import json
import zipfile
import numpy as np
from datetime import datetime

# enable importing from current dir when running with Blender
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from defmo import render
from defmo import utils


# restart with Blender if necessary
args = render.ensure_blender(r"C:\Program Files\Blender Foundation\Blender 2.91\blender.exe")

n_sequences = int(args[0]) if len(args) else 5
out_dir = args[1] if len(args) > 1 else "."

objs = utils.ZipLoader("data/ShapeNetCore.v2.zip", "*.obj")
texs = utils.ZipLoader("data/textures.zip", "*/textures_train/*.jpg")

p = dict(
    resolution=(320, 240),
    n_frames=24,
    blurs=[(0, 10), (-11, -1)],
    z_range=(-8, -3),
    delta_z=1,
    delta_xy=(1, 3),
    max_rot=np.pi / 8,
)


render.init(p["n_frames"], p["resolution"])
frustum = render.Frustum(p["z_range"], p["resolution"])

time = datetime.now()
filename = time.strftime(f"fmo_{len(p['blurs'])}_{p['n_frames']}_%y%m%d%H%M%S.zip")

with zipfile.ZipFile(os.path.join(out_dir, filename), "w") as zip:
    zip.comment = json.dumps(p).encode()
    for i in range(n_sequences):
        obj = objs.get_random(balance_subdirs=True)
        tex = texs.get_random()
        loc = frustum.gen_point_pair(p["delta_z"], p["delta_xy"])
        rot_start = np.random.rand(3) * 2 * np.pi
        rot_end = rot_start + p["max_rot"] * (np.random.rand(3) * 2 - 1)
        name = f"{i:04}.webp"
        with objs.as_tempfile(obj) as objf, texs.as_tempfile(tex) as texf:
            with zip.open(name, "w") as out:
                render.render(out, objf, texf, loc, (rot_start, rot_end), p["blurs"])
        zip.getinfo(name).comment = json.dumps({"obj": obj, "tex": tex}).encode()

duration = datetime.now() - time
print(
    f"Rendered {n_sequences} sequences in {duration}, "
    f"{duration.total_seconds() / n_sequences:.2f}s per sequence"
)
