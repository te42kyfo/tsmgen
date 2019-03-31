from kernel import Kernel
from benchmark import *
import numpy as np
import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(6)

params = [(4, 4), (64, 8)]

color = 0
for p in params:
    globalAtomicPerf = []
    localAtomicPerf = []
    storePerf = []
    nonePerf = []

    globalAtomicKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="globalAtomic")
    localAtomicKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="localAtomic")
    storeKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="store")
    noneKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="none")
    sizes = []

    for k in range(10, 39, 1):
        KK = int(2**(k / 2) * 1024 // p[0])
        globalAtomicPerf.append(benchKernel(globalAtomicKernel, KK, 10))
        localAtomicPerf.append(benchKernel(localAtomicKernel, KK, 10))
        storePerf.append(benchKernel(storeKernel, KK, 10))
        nonePerf.append(benchKernel(noneKernel, KK, 10))
        sizes.append(KK)
    ax.plot(
        sizes,
        np.array(globalAtomicPerf) / np.array(nonePerf),
        "-d",
        color="C" + str(color),
        label="global atomic")
    ax.plot(
        sizes,
        np.array(localAtomicPerf) / np.array(nonePerf),
        "-o",
        color="C" + str(color),
        label="local atomic")
    ax.plot(
        sizes,
        np.array(storePerf) / np.array(nonePerf),
        ":x",
        color="C" + str(color),
        label="store")

    color += 1
    print()

ax.set_xticks( sizes)

ax.set_xscale("log")
ax.set_yticks([0.5, 1])
ax.yaxis.grid(True)
ax.set_xlabel("K")


plt.legend()
plt.savefig("reduction_perf.png", dpi=300, pad_inches=0.0, bbox_inches="tight")
