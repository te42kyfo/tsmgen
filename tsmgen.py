#!/bin/python


import pycuda
import pycuda.autoinit

import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule


def genCode(M, N, TM, TN, blockSize):

    if M % TM != 0 or N % TN != 0:
        print("Tilesize does not match: M % TM != 0 or N % TN != 0\n")
        return ""

    name = "tsmttsm_{0}_{1}_{2}_{3}".format(M, N, TM, TN)

    mthreads = M // TM
    nthreads = N // TN
    threadsPerSlice = mthreads * nthreads

    dtype = "double"
    code = "void __global__ {0} ({1}* A, {1}* B, {1}* C, int K) {{\n".format(name, dtype)
    code += "    int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"
    code += "    int sliceId = tidx / {};\n".format(threadsPerSlice)
    code += "    int midx = tidx % {};\n".format(mthreads)
    code += "    int nidx = (tidx / {})  % {};\n".format(mthreads, nthreads)
    code += "\n"

    code += "    if( tidx >= (gridDim.x*blockDim.x/{0})*{0}) return;\n\n".format(threadsPerSlice)

    for m in range(0, TM):
        code += "    {} ".format(dtype)
        first = True
        for n in range(0, TN):
            if not first:
                code += ", "
            first = False
            code += "tS{}_{}=0".format(m, n)

        code += ";\n"
    code += "\n"

    code += "    for( int idx = sliceId; idx < K; idx += (gridDim.x*{})/{}){{\n".format(blockSize, threadsPerSlice)

    for m in range(0, TM):
        for n in range(0, TN):
            code += "        tS{0}_{1} += A[idx*{2} + midx*{4} + {0}] * B[idx*{3} + nidx*{5} + {1}];\n".format(
                m, n, M, N, TM, TN)

    code += "    }\n"

    for m in range(0, TM):
        for n in range(0, TN):
            code += "    atomicAdd(C + (midx*{0} + {2}) * {4} +  (nidx*{1} + {3}), tS{2}_{3});\n".format(TM, TN, m, n, N)

    code += "}\n"
    return code, name


def compileAndRun(source, name, M, N, K):
    print(source)
    mod = SourceModule(source, arch="sm_60")

    kernel_function = mod.get_function(name)

    A = numpy.ones((K, M), dtype = numpy.float64)
    B = numpy.ones((K, N), dtype = numpy.float64)
    C = numpy.ones((M, N), dtype = numpy.float64)

    kernel_function(drv.In(A), drv.In(B), drv.Out(C), numpy.int32(K), block=(256, 1, 1), grid=(2, 1))

    print(C)
    print()


compileAndRun(*genCode(1, 1, 1, 1, 256), 1, 1, 20123)
compileAndRun(*genCode(20, 8, 4, 4, 256), 20, 8, 20123)
