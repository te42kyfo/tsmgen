#!/bin/python

from kernel import Kernel
from predict import *
from benchmark import *
import numpy as np
import pycuda.driver as drv
import pycuda

import sys
from subprocess import run, PIPE

import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt



#for MN in range(64, 65):
#    for TM in range(1, 17):
#        for TN in range(1, 17):
#            if MN % TM != 0 or MN % TN != 0:
#                continue
#            print( str(MN) + "  " + str(TM) + " " + str(TN) + ":  ", end="")

kernels = []
memPreds0 = []
memPreds1 = []
L1Preds = []
ALUPreds = []
UniPreds = []
flopses = []

for MN in [1, 2, 4, 8, 16, 32, 64]:
    for TM in range(1, 33):
        for TN in range(1, 33):
            if MN % TN != 0 or MN % TM != 0:
                continue
            if TM*TN > 100:
                continue
            kernels.append(Kernel(MN, MN, TM, TN, 256, reduction="none"))

for kernel in kernels:
    memPred = predictMemory(kernel)
    memPreds0.append(memPred[0])
    memPreds1.append(memPred[1])
    L1Pred = predictL1(kernel)
    L1Preds.append(L1Pred)
    ALUPred = predictALU(kernel)
    ALUPreds.append(ALUPred)
    UniPred = predictUniform(kernel)
    UniPreds.append(UniPred)
    print(kernel.name)
    print("Mem lat:   {:7.0f}".format(memPred[0]))
    print("Mem thru:  {:7.0f}".format(memPred[1]))
    print("L1:        {:7.0f}".format(L1Pred))
    print("ALU:       {:7.0f}".format(ALUPred))
    print("Uni:       {:7.0f}".format(UniPred))
    flops = benchKernel(kernel, maxBufferElements // (kernel.M + kernel.N))
    flopses.append(flops)

    print()

fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(8)

plt.ioff()
#ax.bar(np.arange(0, len(kernels)) - 0.225, memPreds0, width=0.15)
#ax.bar(np.arange(0, len(kernels)) - 0.075, memPreds1, width=0.15)
#ax.bar(np.arange(0, len(kernels)) + 0.075, L1Preds, width=0.15)
#ax.bar(np.arange(0, len(kernels)) + 0.225, ALUPreds, width=0.15)
ax.bar(np.arange(0, len(kernels)) + 0, UniPreds, width=0.35)

#ax.hlines([min(z) for z in zip(memPreds0, memPreds1, L1Preds, ALUPreds)],
#          np.arange(0, len(kernels)) + -0.3,
#          np.arange(0, len(kernels)) + 0.425)

ax.plot(np.arange(0, len(kernels)), flopses, "X", color="black")

ax.set_ylim((0, 8000))
ax.set_xlim((-1, len(kernels)))

ax.set_xticks(np.arange(0, len(kernels)))
ax.set_xticklabels([kernel.name for kernel in kernels],
                   rotation=45,
                   ha="right",
                   rotation_mode="anchor")
plt.savefig("perf.png", dpi=300, pad_inches=0.0, bbox_inches="tight")
