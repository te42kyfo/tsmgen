#!/bin/python

import pycuda
import pycuda.autoinit

import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule


class Kernel:
    def __init__(self, M, N, TM, TN, blockSize):
        self.M = M
        self.N = N
        self.TM = TM
        self.TN = TN
        self.name = "tsmttsm_{0}_{1}_{2}_{3}".format(M, N, TM, TN)
        self.blockSize = blockSize
        text = ""
        self.mod = None

    def get_function(self):
        self.mod = SourceModule(self.text, arch="sm_60")
        return self.mod.get_function(self.name)

    def run(self, A, B, C, K):
        minimumBlockCount = (self.M * self.N // self.TM // self.TN - 1) // self.blockSize + 1
        deviceBlockCount = 24

        self.get_function()(
            drv.In(A),
            drv.In(B),
            drv.Out(C),
            numpy.int32(K),
            block=(256, 1, 1),
            grid=(max(deviceBlockCount, minimumBlockCount), 1))


def genCode(M, N, TM, TN, blockSize):

    if M % TM != 0 or N % TN != 0:
        print("Tilesize does not match: M % TM != 0 or N % TN != 0\n")
        return ""

    kernel = Kernel(M, N, TM, TN, blockSize)

    mthreads = M // TM
    nthreads = N // TN
    threadsPerSlice = mthreads * nthreads

    dtype = "double"
    kernel.text = "void __global__ {0} ({1}* A, {1}* B, {1}* C, int K) {{\n".format(kernel.name, dtype)
    kernel.text += "    int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"
    kernel.text += "    int sliceId = tidx / {};\n".format(threadsPerSlice)
    kernel.text += "    int midx = tidx % {};\n".format(mthreads)
    kernel.text += "    int nidx = (tidx / {})  % {};\n".format(mthreads, nthreads)
    kernel.text += "\n"

    kernel.text += "    if( tidx >= (gridDim.x*blockDim.x/{0})*{0}) return;\n\n".format(threadsPerSlice)

    for m in range(0, TM):
        kernel.text += "    {} ".format(dtype)
        first = True
        for n in range(0, TN):
            if not first:
                kernel.text += ", "
            first = False
            kernel.text += "tS{}_{}=0".format(m, n)

        kernel.text += ";\n"
    kernel.text += "\n"

    kernel.text += "    for( int idx = sliceId; idx < K; idx += (gridDim.x*{})/{}){{\n".format(
        blockSize, threadsPerSlice)

    for m in range(0, TM):
        for n in range(0, TN):
            kernel.text += "        tS{0}_{1} += A[idx*{2} + midx*{4} + {0}] * B[idx*{3} + nidx*{5} + {1}];\n".format(
                m, n, M, N, TM, TN)

    kernel.text += "    }\n"

    for m in range(0, TM):
        for n in range(0, TN):
            kernel.text += "    atomicAdd(C + (midx*{0} + {2}) * {4} +  (nidx*{1} + {3}), tS{2}_{3});\n".format(
                TM, TN, m, n, N)

    kernel.text += "}\n"
    return kernel


def compileAndRun(kernel, K):
    print(kernel.text)

    A = numpy.ones((K, kernel.M), dtype=numpy.float64)
    B = numpy.ones((K, kernel.N), dtype=numpy.float64)
    C = numpy.ones((kernel.M, kernel.N), dtype=numpy.float64)

    kernel.run(A, B, C, K)

    print(C)
    print()


compileAndRun(genCode(4, 4, 1, 1, 256), 10000000)
compileAndRun(genCode(4, 4, 2, 2, 256), 10000000)
compileAndRun(genCode(4, 4, 1, 4, 256), 10000000)
compileAndRun(genCode(4, 4, 4, 1, 256), 10000000)
