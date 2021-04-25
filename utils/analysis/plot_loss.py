import sys
import pickle

import matplotlib.pyplot as plt

with open(sys.argv[1], "rb") as f:
    epochs = pickle.load(f)

train = [
    (epoch + i / len(epochs[epoch]["train"]) - 1, val)
    for epoch in epochs
    for i, val in enumerate(epochs[epoch]["train"])
]
valid = [(epoch, epochs[epoch]["valid"]) for epoch in epochs]

train.sort(), valid.sort()

plt.plot(*zip(*train), label="train (mean of 100 inputs)")
plt.plot(*zip(*valid), label="valid", marker="o")

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()
plt.minorticks_on()

plt.show()
