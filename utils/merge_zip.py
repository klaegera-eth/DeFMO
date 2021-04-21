import os
import sys
import zipfile


zips = sys.argv[1:]
out_path = os.path.splitext(zips[0])[0] + "_merged.zip"
with zipfile.ZipFile(out_path, "w") as out:
    seq_n = 0
    for zip in zips:
        with zipfile.ZipFile(zip) as f:
            if not out.comment:
                out.comment = f.comment
            for file in f.filelist:
                name = f"{seq_n:04}.webp"
                out.writestr(name, f.read(file))
                out.getinfo(name).comment = file.comment
                seq_n += 1
