from kernel import Kernel
from benchmark import *
import numpy as np
import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_figwidth(6)
fig.set_figheight(4)

params = [(4, 2, 2, 1024), (64, 8, 8, 256)]

color = 0
for p in params:
    globalAtomicPerf = []
    localAtomicPerf = []
    storePerf = []
    nonePerf = []

    globalAtomicKernel = Kernel(p[0], p[0], p[1], p[2], p[3], reduction="globalAtomic")
    localAtomicKernel = Kernel(p[0], p[0], p[1], p[2],  p[3], reduction="localAtomic")
    storeKernel = Kernel(p[0], p[0], p[1], p[1], p[3], reduction="store")
    noneKernel = Kernel(p[0], p[0], p[1], p[2], p[3], reduction="none")
    sizes = []

    for k in range(1, 39, 1):
        KK = int(2**(k / 2) * 1024 // p[0])
        globalAtomicPerf.append(benchKernel(globalAtomicKernel, KK, 10)[2])
        localAtomicPerf.append(benchKernel(localAtomicKernel, KK, 10)[2])
        storePerf.append(benchKernel(storeKernel, KK, 10)[2])
        nonePerf.append(benchKernel(noneKernel, KK, 10)[2])
        sizes.append(KK)

    print(globalAtomicPerf)
    print(localAtomicPerf)
    print(nonePerf)
    ax.plot(
        sizes[1:],
        np.array(globalAtomicPerf[1:]) / np.array(nonePerf[1:]),
        "-d",
        color="C" + str(color),
        label= "M,N={}, tile size={} global atomic".format(p[0], p[1]))
    
    ax.plot(
        sizes[1:],
        np.array(localAtomicPerf[1:]) / np.array(nonePerf[1:]),
        "-o",
        color="C" + str(color),
        label= "M,N={}, tile size={} local atomic".format(p[0], p[1]))
    #ax.plot(
    #    sizes,
    #    np.array(storePerf) / np.array(nonePerf),
    #    ":x",
    #    color="C" + str(color),
    #    label= noneKernel.name + " store")

    color += 1
    print()

ax.set_xticks( sizes)

ax.set_xscale("log")
ax.set_yticks([0, 0.5, 0.9, 1])
ax.yaxis.grid(True)
ax.set_xlabel("K")
#ax.set_ylim((0, 1.2))

plt.legend()
plt.savefig("reduction_perf.pdf", dpi=300, pad_inches=0.0, bbox_inches="tight")
