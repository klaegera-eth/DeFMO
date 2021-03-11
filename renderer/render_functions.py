import bpy
import os
import math
import tempfile
import numpy as np
from PIL import Image
from zipfile import ZipFile
from contextlib import contextmanager
from collections import defaultdict
import random
import fnmatch


def init(frames, resolution, mblur=40, env_light=(1, 1, 1)):
    scene = bpy.context.scene

    # output settings
    scene.frame_start = scene.frame_current = 0
    scene.frame_end = frames - 1
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

    # motion blur parameters
    scene.eevee.motion_blur_position = "START"
    scene.eevee.motion_blur_steps = mblur


def render(output, obj, tex, loc_from, loc_to, rot_from, rot_to, blurs=[(0, -1)]):
    scene = bpy.context.scene

    # load object
    bpy.ops.import_scene.obj(filepath=obj)
    obj = bpy.context.selected_objects[0]

    # load texture
    if tex:
        tex = load_image(tex)
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

        # render frames
        scene.eevee.use_motion_blur = False
        scene.render.filepath = os.path.join(tmp, "frame")
        bpy.ops.render.render(animation=True)

        # render blurs
        scene.eevee.use_motion_blur = True
        for i, blur in enumerate(blurs):
            blur_start, blur_end = [b % (scene.frame_end + 1) for b in blur]
            scene.frame_current = blur_start
            scene.eevee.motion_blur_shutter = blur_end - blur_start
            scene.render.filepath = os.path.join(tmp, f"blur{i:04}")
            bpy.ops.render.render(write_still=True)
        scene.frame_current = scene.frame_start

        # pack to webp
        fs = sorted([os.path.join(tmp, f) for f in os.listdir(tmp)])
        Image.open(fs[0]).save(
            output,
            format="webp",
            save_all=True,
            append_images=(Image.open(f) for f in fs[1:]),
            method=4,
            quality=75,
            duration=[1000] * len(blurs) + [33] * (scene.frame_end + 1),
            minimize_size=True,
        )

    # clean up
    bpy.ops.object.delete()
    if tex:
        bpy.data.images.remove(tex)


def load_image(fp):
    try:
        return bpy.data.images.load(os.path.abspath(fp))
    except:
        # try to load image with PIL
        pil = fp if isinstance(fp, Image.Image) else Image.open(fp)
        img = bpy.data.images.new("img", pil.width, pil.height)
        img.pixels[:] = np.asarray(pil.convert("RGBA")).ravel() / 255
        return img


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


class ZipLoader:
    def __init__(self, zip, filter="*[!/]", balance_subdirs=False):
        self.zip = ZipFile(zip)
        self.names = fnmatch.filter(self.zip.namelist(), filter)
        self.tree = None
        if balance_subdirs:
            # create directory tree of zip contents
            dict_tree = lambda: defaultdict(dict_tree)
            self.tree = dict_tree()
            for name in self.names:
                node = self.tree
                for d in name.split("/")[:-1]:
                    node = node[d]
                node[name] = None

    @contextmanager
    def get(self, name):
        _, ext = os.path.splitext(name)
        fd, path = tempfile.mkstemp(suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(self.zip.read(name))
        try:
            yield path
        finally:
            os.remove(path)

    def get_random(self):
        if self.tree:
            # randomly sample at every level of directory tree
            node = self.tree
            while True:
                name = random.choice(list(node.keys()))
                node = node[name]
                if not node:
                    # leaf node
                    return self.get(name)
        return self.get(random.choice(self.names))
