import io
import os
import sys
import json
import zipfile
import numpy as np
from PIL import Image
from datetime import datetime

# enable importing from current dir when running with Blender
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from defmo import render
from defmo import utils

# print to stderr when suppressing Blender output
from defmo.utils import print_stderr as print

# restart with Blender if necessary
args = render.ensure_blender(r"C:\Program Files\Blender Foundation\Blender 2.91\blender.exe", suppress_output=True)


n_sequences = int(args[0]) if len(args) else 5
out_dir = args[1] if len(args) > 1 else "."

print("Loading datasets...")
objs = utils.ZipLoader("data/ShapeNetCore.v2.zip", "*.obj", balance_subdirs=True)
texs = utils.ZipLoader("data/textures.zip", "*/textures_train/*.jpg")

p = dict(
    resolution=(320, 240),
    n_frames=24,
    blurs=[(0, 10), (-11, -1)],
    z_range=(-8, -3),
    delta_z=1,
    delta_xy=(1, 3),
    max_rot=np.pi / 8,
    min_alpha=255 / 3,
)


render.init(p["n_frames"], p["resolution"])
frustum = render.Frustum(p["z_range"], p["resolution"])

time = datetime.now()
filename = time.strftime(f"fmo_{len(p['blurs'])}_{p['n_frames']}_%y%m%d%H%M%S.zip")

with zipfile.ZipFile(os.path.join(out_dir, filename), "w") as zip:

    # write parameters to zip
    zip.comment = json.dumps(p).encode()

    for seq_n in range(n_sequences):
        while True:

            # randomly generate render parameters
            obj = objs.get_random()
            tex = texs.get_random()
            loc = frustum.gen_point_pair(p["delta_z"], p["delta_xy"])
            rot_start = np.random.rand(3) * 2 * np.pi
            rot_end = rot_start + p["max_rot"] * (np.random.rand(3) * 2 - 1)

            with io.BytesIO() as out:
                with objs.as_tempfile(obj) as objf, texs.as_tempfile(tex) as texf:
                    render.render(out, objf, texf, loc, (rot_start, rot_end), p["blurs"])

                # check if highest alpha value is at least min_alpha
                if p["min_alpha"]:
                    alpha = np.array(Image.open(out))[:, :, 3]
                    if np.max(alpha) < p["min_alpha"]:
                        print("Alpha too low, restarting...")
                        continue

                # save render to zip
                name = f"{seq_n:04}.webp"
                zip.writestr(name, out.getvalue())
                zip.getinfo(name).comment = json.dumps({"obj": obj, "tex": tex}).encode()

            break

        # print progress
        duration = datetime.now() - time
        print(f"{seq_n + 1}/{n_sequences} - {duration} = {duration.total_seconds() / (seq_n + 1):.2f}s/seq")

        # check if "stop" file exists to abort rendering early
        if os.path.exists(os.path.join(out_dir, "stop")):
            print("Stop file found")
            break
