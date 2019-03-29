from kernel import Kernel
from benchmark import *
import numpy as np
import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(8)

atomicPerf = []
nonePerf = []
localPerf = []

params = [(4, 4), (8, 4), (16, 8), (32, 8), (64, 8)]

for p in params:
    atomicKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="globalAtomic")
    localKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="localAtomic")
    noneKernel = Kernel(p[0], p[0], p[1], p[1], 256, reduction="none")

    KK = maxBufferElements / p[0] / 2

    nonePerf.append(benchKernel(noneKernel, KK, 2))
    atomicPerf.append(benchKernel(atomicKernel, KK, 2))
    localPerf.append(benchKernel(localKernel, KK, 2))

    print()

ax.plot(np.array(atomicPerf) / np.array(nonePerf), "x-")
ax.plot(np.array(localPerf) / np.array(nonePerf), "x-")

#ax.set_xticks( [2**i + 3 for i in range(10,23)])
ax.set_xscale("log")

ax.set_yticks([0.5, 1, 2.0])
ax.yaxis.grid(True)

plt.savefig("reduction_perf.png", dpi=300, pad_inches=0.0, bbox_inches="tight")
