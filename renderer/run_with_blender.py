import sys
import subprocess

blender = "C:\\Program Files\\Blender Foundation\\Blender 2.91\\blender.exe"

if len(sys.argv) < 2:
    print("missing argument")
else:
    subprocess.run([blender, "--background", "--python", sys.argv[1], "--"] + sys.argv[2:])
