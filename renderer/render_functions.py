import bpy
import os
import math
import tempfile
import numpy as np
from PIL import Image


def init(frames, resolution, mblur=40, env_light=(1, 1, 1)):
    scene = bpy.context.scene

    # output settings
    scene.frame_end = frames
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    # canonical scene
    scene.camera.location = 0, 0, 0
    scene.camera.rotation_euler = 0, 0, 0
    scene.objects["Light"].location = 0, 0, 0

    # environment lighting
    bpy.context.scene.world.use_nodes = False
    bpy.context.scene.world.color = env_light

    # remove default cube
    bpy.ops.object.delete()

    # create material for texture
    mat = bpy.data.materials.new("Texture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    mat.node_tree.links.new(tex.outputs[0], nodes["Principled BSDF"].inputs[0])

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
    scene.eevee.motion_blur_steps = mblur


def render(output, obj, img, tex, loc_from, loc_to, rot_from, rot_to):
    scene = bpy.context.scene

    # load object
    bpy.ops.import_scene.obj(filepath=obj)
    obj = bpy.context.selected_objects[0]

    # load background image
    img = bpy.data.images.load(os.path.abspath(img))
    scene.node_tree.nodes["Image"].image = img

    # load texture
    if tex:
        tex = bpy.data.images.load(os.path.abspath(tex))
        mat = bpy.data.materials["Texture"]
        mat.node_tree.nodes["Image Texture"].image = tex
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.editmode_toggle()
        bpy.ops.uv.cube_project(scale_to_bounds=True)
        bpy.ops.object.editmode_toggle()

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

    # render to temp dir then pack to webp
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
            output,
            format="webp",
            save_all=True,
            append_images=(Image.open(f) for f in fs[:1] + fs[2:]),
            method=4,
            quality=75,
            minimize_size=True,
        )

    # clean up
    bpy.ops.object.delete()
    bpy.data.images.remove(img)
    if tex:
        bpy.data.images.remove(tex)


def calc_frustum(max_radius=0.5, dead_zone=0.05, focal_length=50, sensor_size=36):
    tan = sensor_size / focal_length / 2
    alpha = math.atan(tan) * (1 - dead_zone)
    offset = max_radius / math.sin(alpha)
    return math.tan(alpha), offset


def gen_frustum_point(z_range, res, tan, offset):
    x, y, z = np.random.rand(3)
    z = z_range[0] + (z_range[1] - z_range[0]) * z
    x = (x * 2 - 1) * tan * (z + offset)
    y = (y * 2 - 1) * tan * (z * res[1] / res[0] + offset)
    return x, y, z
