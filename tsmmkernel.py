import pycuda
import math
import pycuda.autoinit
import sys
from subprocess import run, PIPE

import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule


class TSMMKernel:

    def __init__(self, M, N, TN, blockSize, unroll=1):
        self.M = M
        self.N = N
        self.TN = TN
        self.name = "tsmm_{0}_{1}_{2}".format(M, N, TN)
        self.blockSize = blockSize
        self.text = ""
        self.mod = None
        self.multiprocessor_count = None
        self.function = None

        dtype = "double"

        nthreads = (N - 1) // TN + 1

        self.text = "void __global__ __launch_bounds__({2}) {0} ({1}* A, {1}* B, {1}* C, int64_t K) {{\n".format(
            self.name, dtype, self.blockSize)
        self.text += "  int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"
        self.text += "  const int sliceId = tidx / {};\n".format(nthreads)
        self.text += "  const int nidx = tidx % {};\n".format(nthreads)
        self.text += "  const int gridStride = (gridDim.x*{})/{};\n\n".format(
            blockSize, nthreads, unroll)

        # iteration loop
        self.text += "  int64_t idx;\n"
        self.text += "  for(idx = sliceId; idx < K-gridStride*{1}; idx += gridStride * {0}){{\n".format(
            unroll, unroll - 1)

        self.text += "    {} ".format(dtype)
        first = True
        for u in range(0, unroll):
            for n in range(0, TN):
                if not first:
                    self.text += ", "
                first = False
                self.text += "ts{}_{} = 0".format(n, u)
        self.text += ";\n\n"

        for tn in range(0, TN):
            for m in range(0, M):
                for u in range(0, unroll):
                    if (tn + 1) * nthreads > N:
                        self.text += "    if( nidx < {} )\n  ".format(N - tn * nthreads)

                    self.text += "    ts{0}_{5} += A[(idx+{5}*gridStride)*{1} + {2}] * C[{2}*{3} + {0}*{4} + nidx];\n".format(
                        tn, M, m, N, nthreads, u)

        for n in range(0, TN):
            for u in range(0, unroll):
                if (n + 1) * nthreads > N:
                    self.text += "    if( nidx < {} )\n  ".format(N - n * nthreads)
                self.text += "    B[(idx+{3}*gridStride)*{0} +  {1}*{2} + nidx] = ts{1}_{3};\n".format(
                    N, n, nthreads, u)

        self.text += "  } \n"

        # remainder iteration loop
        self.text += "  for(; idx < K; idx += gridStride){\n"
        self.text += "    {} ".format(dtype)
        first = True
        for n in range(0, TN):
            if not first:
                self.text += ", "
            first = False
            self.text += "ts{}_{} = 0".format(n, u)
        self.text += ";\n\n"

        for tn in range(0, TN):
            for m in range(0, M):
                if (tn + 1) * nthreads > N:
                    self.text += "    if( nidx < {} )\n  ".format(N - tn * nthreads)

                self.text += "    ts{0}_{5} += A[(idx)*{1} + {2}] * C[{2}*{3} + {0}*{4} + nidx];\n".format(
                    tn, M, m, N, nthreads, u)

        for n in range(0, TN):
            if (n + 1) * nthreads > N:
                self.text += "    if( nidx < {} )\n  ".format(N - n * nthreads)
            self.text += "    B[(idx)*{0} +  {1}*{2} + nidx] = ts{1}_{3};\n".format(
                N, n, nthreads, u)

        self.text += "  } \n"
        self.text += "} \n\n"

    def run(self, A, B, C, K, blocksPerMP=-1):

        if self.multiprocessor_count is None:
            self.multiprocessor_count = drv.Context.get_device().get_attributes()[
                drv.device_attribute.MULTIPROCESSOR_COUNT]
        if self.mod is None:
            self.mod = SourceModule(self.text, arch="sm_70", options=["-lineinfo"])
        if self.function is None:
            self.function = self.mod.get_function(self.name)
            self.function.prepare(('P', 'P', 'P', numpy.int64))

        if self.blockSize * self.function.num_regs > pycuda.tools.DeviceData().registers:
            return

        tb_per_mp = self.estimateBlockCount(self.function.num_regs) * 100

        threadsPerSlice = math.ceil(self.N / self.TN)

        block = (self.blockSize, 1, 1)
        if blocksPerMP == -1:
            grid = (self.multiprocessor_count * tb_per_mp,)
        else:
            grid = (80,)

        self.function.prepared_call(grid, block, A, B, C, numpy.int64(K))

    def estimateBlockCount(self, registers):
        return min(64 // (self.blockSize // 32),
                   65536 // (max(32, registers) * self.blockSize))
