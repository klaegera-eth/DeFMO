import os, sys
import tempfile
from PIL import Image


def ensure_blender(blender=None, suppress_output=False):
    try:
        global bpy
        import bpy

        return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    except ImportError:
        if blender:
            import subprocess

            print("Restarting with Blender...")
            sys.stdout.flush()
            sys.stderr.flush()
            proc = subprocess.Popen(
                [blender, "--background", "--python", sys.argv[0], "--"] + sys.argv[1:],
                stdout=subprocess.DEVNULL if suppress_output else None,
                stderr=subprocess.PIPE,
            )
            for line in proc.stderr:
                sys.stdout.write(line.decode())
            sys.exit()
        else:
            sys.exit("Failed to import bpy. Please run with Blender.")


def init(frames, resolution, mblur=40, env_light=(1, 1, 1)):
    ensure_blender()
    scene = bpy.context.scene

    # output settings
    scene.frame_start = scene.frame_current = 0
    scene.frame_end = frames - 1
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    # canonical camera
    scene.camera.location = 0, 0, 0
    scene.camera.rotation_euler = 0, 0, 0

    # environment lighting
    bpy.context.scene.world.use_nodes = False
    bpy.context.scene.world.color = env_light

    # remove default objects
    bpy.ops.object.delete()
    bpy.data.objects.remove(scene.objects["Light"])

    # create material for texture
    mat = bpy.data.materials.new("Texture")
    mat.use_fake_user = True
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    mat.node_tree.links.new(tex.outputs[0], nodes["Principled BSDF"].inputs[0])

    # motion blur parameters
    scene.eevee.motion_blur_position = "START"
    scene.eevee.motion_blur_steps = mblur


def render(output, obj, tex, loc, rot, blurs=[(0, -1)]):
    ensure_blender()
    scene = bpy.context.scene

    # load object
    bpy.ops.import_scene.obj(filepath=obj)
    obj = bpy.context.selected_objects[0]

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
    obj.location = loc[0]
    obj.rotation_euler = rot[0]
    obj.keyframe_insert("location")
    obj.keyframe_insert("rotation_euler")

    # final position
    obj.location = loc[1]
    obj.rotation_euler = rot[1]
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
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for block in collection:
            if not block.users:
                collection.remove(block)
