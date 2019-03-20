#!/bin/python

from kernel import Kernel
import numpy as np
import pycuda.driver as drv

maxBufferElements = 128 * 1024 * 1024

A_gpu = drv.mem_alloc(maxBufferElements * 8)
B_gpu = drv.mem_alloc(maxBufferElements * 8)
C_gpu = drv.mem_alloc(128 * 128 * 8)


def benchKernel(kernel, K):

    kernel.run(A_gpu, B_gpu, C_gpu, K)

    def timeKernel():
        start = drv.Event()
        end = drv.Event()

        start.record()
        kernel.run(A_gpu, B_gpu, C_gpu, K)
        end.record()
        end.synchronize()
        return end.time_since(start)

    dts = [timeKernel() for i in range(0, 1)]

    dt = min(dts)

    bw = (kernel.M * K + kernel.N * K) * 8 / dt / 10**6
    flops = kernel.M * K * kernel.N * 2 / dt / 10**6

    print("{:6.3f}  {:6.1f}  {:6.1f}".format(dt, bw, flops))


#for MN in range(64, 65):
#    for TM in range(1, 17):
#        for TN in range(1, 17):
#            if MN % TM != 0 or MN % TN != 0:
#                continue
#            print( str(MN) + "  " + str(TM) + " " + str(TN) + ":  ", end="")

kernel = Kernel(64, 64, 8, 8, 256)

warpAddresses = [kernel.genAddresses(warpLane) for warpLane in range(0, 32)]

L1CLs = [set([warpAddresses[warpLane][load][1] // 128 for warpLane in range(0, 32)])
                 for load in range(len(warpAddresses[0]))]

L1cycles = sum([len(load) for load in L1CLs])

print(warpAddresses)
print()
print(L1CLs)

print(L1cycles)
print(8*8)

