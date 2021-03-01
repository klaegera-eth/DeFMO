import bpy
import os, tempfile
from PIL import Image


def init(frames, resolution):
    scene = bpy.context.scene

    # output settings
    scene.frame_end = frames
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    # default camera position
    scene.camera.location = 0, 0, 4
    scene.camera.rotation_euler = 0, 0, 0

    # remove default cube
    bpy.ops.object.delete()

    # configure compositor nodes
    # bg image -> scale to fit -> overlay blurred object
    scene.use_nodes = True
    nodes = scene.node_tree.nodes
    img = nodes.new("CompositorNodeImage")
    scale = nodes.new("CompositorNodeScale")
    scale.frame_method = "CROP"
    scale.space = "RENDER_SIZE"
    alpha = nodes.new("CompositorNodeAlphaOver")
    fout = nodes.new("CompositorNodeOutputFile")

    links = scene.node_tree.links
    links.new(img.outputs[0], scale.inputs[0])
    links.new(scale.outputs[0], alpha.inputs[1])
    links.new(scale.outputs[0], fout.inputs[0])
    links.new(nodes["Render Layers"].outputs[0], alpha.inputs[2])
    links.new(alpha.outputs[0], nodes["Composite"].inputs[0])

    # motion blur over all frames for FMO
    scene.eevee.motion_blur_position = "START"
    scene.eevee.motion_blur_shutter = scene.frame_end - 1
    scene.eevee.motion_blur_steps = 40


def render(path, img, obj, loc_from, loc_to, rot_from, rot_to):
    scene = bpy.context.scene

    # load img and obj
    img = bpy.data.images.load(os.path.abspath(img))
    scene.node_tree.nodes["Image"].image = img
    bpy.ops.import_scene.obj(filepath=obj)
    obj = bpy.context.selected_objects[0]

    # starting position
    obj.location = loc_from
    obj.rotation_euler = rot_from
    obj.keyframe_insert("location")
    obj.keyframe_insert("rotation_euler")

    # final position
    obj.location = loc_to
    obj.rotation_euler = rot_to
    obj.keyframe_insert("location", frame=scene.frame_end)
    obj.keyframe_insert("rotation_euler", frame=scene.frame_end)

    # linear movement
    for f in obj.animation_data.action.fcurves:
        for k in f.keyframe_points:
            k.interpolation = "LINEAR"

    # render frames to temp dir
    with tempfile.TemporaryDirectory() as tmp:
        scene.render.filepath = os.path.join(tmp, "img")
        scene.node_tree.nodes["File Output"].base_path = tmp

        # render frames (no bg, no blur)
        scene.use_nodes = False
        scene.eevee.use_motion_blur = False
        bpy.ops.render.render(animation=True)

        # render FMO
        scene.use_nodes = True
        scene.eevee.use_motion_blur = True
        bpy.ops.render.render(write_still=True)

        # pack to webp
        fs = [os.path.join(tmp, f) for f in os.listdir(tmp)]
        Image.open(fs[1]).save(
            path + ".webp",
            save_all=True,
            append_images=(Image.open(f) for f in fs[:1] + fs[2:]),
            method=4,
            quality=75,
            minimize_size=True,
        )

    # clean up
    bpy.ops.object.delete()
    bpy.data.images.remove(img)


import time


init(30, (320, 240))

st = time.time()

render(
    "C:/tmp/a",
    "data/vot/seq/shaking/00000001.jpg",
    "data/ShapeNetCore.v2/03261776/2b28e2a5080101d245af43a64155c221/models/model_normalized.obj",
    (-0.25, -0.25, -0.25),
    (0.25, 0.25, 0.25),
    (0, 0, 0),
    (1, 1, 1),
)

render(
    "C:/tmp/b",
    "data/vot/seq/crossing/00000001.jpg",
    "data/ShapeNetCore.v2/02942699/3d18881b51009a7a8ff43d2d38ae15e1/models/model_normalized.obj",
    (-0.25, -0.25, -0.25),
    (0.25, 0.25, 0.25),
    (0, 0, 0),
    (1, 1, 1),
)

print((time.time() - st) / 2)
