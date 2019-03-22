#!/bin/python

from kernel import Kernel
import numpy as np
import pycuda.driver as drv
import pycuda

import sys
from subprocess import run, PIPE

import matplotlib
matplotlib.use("SVG")

import matplotlib.pyplot as plt

maxBufferElements = 128 * 1024 * 1024

A_gpu = drv.mem_alloc(maxBufferElements * 8)
B_gpu = drv.mem_alloc(maxBufferElements * 8)
C_gpu = drv.mem_alloc(128 * 128 * 8)


def printSASS(code):
    cubin = pycuda.compiler.compile(code, options=["-w", "-std=c++11"], arch="sm_70")

    run(["echo \"" + code + "\" >> temp.cubin"], stdout=PIPE, shell=True)

    newFile = open("temp.cusbin", "wb")
    newFile.write(cubin)
    newFile.close()

    result = run(["nvdisasm -c   temp.cusbin"], stdout=PIPE, shell=True)

    print(len(result.stdout.decode("utf-8").split("\n")))

    print(result.stdout.decode("utf-8"))

    newFile = open("temp.disasm", "wb")
    newFile.write(result.stdout)
    newFile.close()


def benchKernel(kernel, K):
    def timeKernel():
        start = drv.Event()
        end = drv.Event()

        start.record()
        kernel.run(A_gpu, B_gpu, C_gpu, K)
        end.record()
        end.synchronize()
        return end.time_since(start)

    dts = [timeKernel() for i in range(0, 10)]

    dt = min(dts)

    bw = (kernel.M * K + kernel.N * K) * 8 / dt / 10**6
    flops = kernel.M * K * kernel.N * 2 / dt / 10**6

    print("{:5.2f} {:6.0f} {:6.0f}".format(dt, bw, flops))
    return flops


#for MN in range(64, 65):
#    for TM in range(1, 17):
#        for TN in range(1, 17):
#            if MN % TM != 0 or MN % TN != 0:
#                continue
#            print( str(MN) + "  " + str(TM) + " " + str(TN) + ":  ", end="")


def predictL1(kernel):
    warpAddresses = [kernel.genAddresses(warpLane) for warpLane in range(0, 32)]

    L1CLs = [
        set([warpAddresses[warpLane][load][1] // 128 for warpLane in range(0, 32)])
        for load in range(len(warpAddresses[0]))
    ]

    L1cycles = sum([max(2, len(load)) for load in L1CLs])

    return kernel.TM * kernel.TN / max(L1cycles, kernel.TM * kernel.TN) * 1.38 * 2 * 32 * 80


def predictMemory(kernel):
    addresses = [[(l[0], l[1] // 64) for l in kernel.genAddresses(warpLane)]
                 for warpLane in range(0, kernel.blockSize)]

    CLs = set([
        addresses[warpLane][load] for warpLane in range(0, kernel.blockSize)
        for load in range(len(addresses[0]))
    ])

    estRegCount = min(256, (kernel.TM * kernel.TN + kernel.TM + kernel.TN) * 2 + 10)
    estBlockCount = kernel.estimateBlockCount(estRegCount)

    aluInstrPerLoop = 10
    instrExecLatency = max(aluInstrPerLoop * 2,
                           aluInstrPerLoop * estBlockCount * (kernel.blockSize / 32) / 2)

    latencyPred = 80 * estBlockCount * kernel.TM * kernel.TN * kernel.blockSize * 2 * (
        1.38 / (436 + kernel.TN * kernel.TM * 4 + (kernel.TM + kernel.TN) * 2 + instrExecLatency))

    throughputPred = 2 * kernel.TM * kernel.TN * kernel.blockSize / (len(CLs) * 64 / 880e9) / 1e9
    return (latencyPred, throughputPred)


def predictALU(kernel):
    aluInstrPerLoop = 10
    GIts = 1.38 * 80 * 2 * 32 / aluInstrPerLoop
    return GIts * kernel.TM * kernel.TN


kernels = [
    Kernel(1, 1, 1, 1, 256),
    Kernel(4, 4, 1, 1, 256),
    Kernel(4, 4, 2, 2, 256),
    Kernel(4, 4, 4, 1, 256),
    Kernel(4, 4, 4, 4, 256),
    Kernel(8, 8, 1, 1, 256),
    Kernel(8, 8, 4, 4, 256),
    Kernel(8, 8, 8, 1, 256),
    Kernel(8, 8, 8, 8, 256),
    Kernel(32, 32, 1, 1, 256),
    Kernel(32, 32, 4, 4, 256),
    Kernel(32, 32, 8, 1, 256),
    Kernel(32, 32, 8, 8, 256),
    Kernel(32, 32, 10, 10, 256),
    Kernel(64, 64, 2, 2, 256),
    Kernel(64, 64, 8, 8, 256),
    Kernel(64, 64, 10, 10, 256),
    Kernel(64, 64, 16, 1, 256)
]

memPreds0 = []
memPreds1 = []
L1Preds = []
ALUPreds = []
flopses = []

for kernel in kernels:
    memPred = predictMemory(kernel)
    memPreds0.append(memPred[0])
    memPreds1.append(memPred[1])
    L1Pred = predictL1(kernel)
    L1Preds.append(L1Pred)
    ALUPred = predictALU(kernel)
    ALUPreds.append(ALUPred)
    print(kernel.name)
    print("Mem lat:   {:7.0f}".format(memPred[0]))
    print("Mem thru:  {:7.0f}".format(memPred[1]))
    print("L1:        {:7.0f}".format(L1Pred))
    print("ALU:       {:7.0f}".format(ALUPred))
    flops = benchKernel(kernel, maxBufferElements // (kernel.M + kernel.N))
    flopses.append(flops)
    print()



fig, ax = plt.subplots()
fig.set_figwidth(16)
fig.set_figheight(8)


plt.ioff()
ax.bar(np.arange(0, len(kernels)) - 0.225, memPreds0, width=0.15)
ax.bar(np.arange(0, len(kernels)) - 0.075, memPreds1, width=0.15)
ax.bar(np.arange(0, len(kernels)) + 0.075, L1Preds, width=0.15)
ax.bar(np.arange(0, len(kernels)) + 0.225, ALUPreds, width=0.15)

ax.hlines([min(z) for z in zip(memPreds0, memPreds1, L1Preds, ALUPreds)],
           np.arange(0, len(kernels)) + -0.3,
           np.arange(0, len(kernels)) + 0.3)

ax.plot(np.arange(0, len(kernels)), flopses, "X", color="black")

ax.set_ylim((0, 8000))

ax.set_xticks(np.arange(0, len(kernels)))
ax.set_xticklabels( [kernel.name for kernel in kernels], rotation=45, ha="right", rotation_mode="anchor")
plt.savefig("perf.png", dpi=300, pad_inches=0.0, bbox_inches="tight")
