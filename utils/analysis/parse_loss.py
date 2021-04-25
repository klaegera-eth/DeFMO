import sys
import re
import pickle

from collections import defaultdict

epochs = defaultdict(lambda: dict(train=[]))
lre = re.compile("Epoch (\d+)[^V]*(Validation)?.* Loss\( ([.\d]+)/([.\d]+)")

with open(sys.argv[1]) as f:
    for line in f:
        match = lre.search(line)
        if match:
            epoch, valid, loss100, lossall = match.groups()

            if valid:
                epochs[int(epoch)]["valid"] = float(lossall)
            else:
                epochs[int(epoch)]["train"].append(float(loss100))


with open(sys.argv[1] + "_epochs.pkl", "wb") as f:
    pickle.dump(dict(epochs), f)
