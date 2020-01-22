import pycuda
import math
import pycuda.autoinit
import sys
from subprocess import run, PIPE

import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule

cu_complex_string = """#include <cuComplex.h>


__device__ inline double daxpy(double val, double val2, double val3)
{
    return val+val2*val3;
}


__device__ inline cuFloatComplex caxpy(cuFloatComplex val, cuFloatComplex val2, cuFloatComplex val3)
{
    return cuCaddf(val,cuCmulf(val2,val3));
}


__device__ inline cuDoubleComplex zaxpy(cuDoubleComplex val, cuDoubleComplex val2, cuDoubleComplex val3)
{
    double real = val.x + - val2.y*val3.y + val2.x*val3.x;
    double imag = val.y + val2.y*val3.x + val2.x*val3.y;

    return make_cuDoubleComplex(real, imag);
}

__device__ inline cuDoubleComplex atomicAdd(cuDoubleComplex* address, cuDoubleComplex val) {
    double* dadd = reinterpret_cast<double*>(address);
    return make_cuDoubleComplex(atomicAdd(dadd, val.x), atomicAdd(dadd + 1, val.y));
}

\n"""

axpy_func = {"double": "daxpy", "cuFloatComplex": "caxpy", "cuDoubleComplex": "zaxpy"}


class Kernel:

    def __init__(self,
                 M,
                 N,
                 TM,
                 TN,
                 blockSize,
                 unroll=1,
                 reduction="globalAtomic",
                 transposed=True,
                 leapFrog=False,
                 singleAccumulator=False,
                 truncatedIndex=False,
                 noIndex=False,
                 dtype="double"):
        self.M = M
        self.N = N
        self.TM = TM
        self.TN = TN
        self.name = "tsmttsm_{0}_{1}_{2}_{3}".format(M, N, TM, TN)
        self.blockSize = blockSize
        self.text = ""
        self.mod = None
        self.multiprocessor_count = None
        self.function = None
        self.dtype = dtype

        #if M % TM != 0 or N % TN != 0:
        #    print("Tilesize does not match: M % TM != 0 or N % TN != 0\n")

        mthreads = (M - 1) // TM + 1
        nthreads = (N - 1) // TN + 1
        threadsPerSlice = mthreads * nthreads


        if mthreads*nthreads* (16 if dtype == "cuDoubleComplex" else 8) > 48*1024 and reduction=="localAtomic":
            reduction = "globalAtomic"

        minBlockCount = min(16,
                            max(1, 65536 // blockSize // ((TM * TN + TM + TN) * 2 + 30)),
                            64 // (blockSize // 32))

        self.text += cu_complex_string

        if dtype == "cuDoubleComplex" or dtype == "cuFloatComplex":
            self.text += "__device__ inline cuDoubleComplex VALUE(double v) {  return make_cuDoubleComplex(v, 0.0);}\n\n"
            self.text += "\n\n"
        else:
            self.text += "__device__ inline  double VALUE(double v) {  return v;}\n\n"

        self.text += "void __global__ __launch_bounds__({2}) {0} ({1}* A, {1}* B, {1}* C, int64_t K) {{\n".format(
            self.name, dtype, self.blockSize)
        self.text += "  int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"
        self.text += "  int sliceId = tidx / {};\n".format(threadsPerSlice)
        self.text += "  int midx = tidx % {};\n".format(mthreads)
        self.text += "  int nidx = (tidx / {})  % {};\n".format(mthreads, nthreads)
        self.text += "\n"
        if reduction == "localAtomic":
            self.text += "  __shared__ {} blockResults[{}][{}];\n".format(
                dtype, mthreads, nthreads)
            self.text += "  for( int id = threadIdx.x; id <  {}; id += {})\n".format(
                mthreads * nthreads, self.blockSize)
            self.text += "    blockResults[id%{0}][id/{0}] = VALUE(0.0);\n\n".format(
                mthreads)

            self.text += "  __syncthreads();\n"

        self.text += "  if( tidx >= (gridDim.x*blockDim.x/{0})*{0}) return;\n\n".format(
            threadsPerSlice)

        for m in range(0, TM):
            self.text += "  {} ".format(dtype)
            first = True
            for n in range(0, TN):
                if not first:
                    self.text += ", "
                first = False
                self.text += "tS{}_{} = VALUE(0.0)".format(m, n)

            self.text += ";\n"
        self.text += "\n"

        self.text += "  int gridStride = (gridDim.x*{})/{};\n".format(
            blockSize, threadsPerSlice)

        def generateLoad(name, array, dtype, X, TX, x, u, tileIdx, xthreads, idx):
            loadText = ""
            loadText += "{0} {1}_{2}_{3} = ".format(dtype, name, x, u)

            if not transposed:
                ldgText = "__ldg( {0} + ({6} {7} +{4}*gridStride)*{2}  + {5}*{3} + {1})".format(
                    array, x, X, TX, u, tileIdx, idx if not noIndex else 0,
                    " / (1024*5000)" if truncatedIndex else "")
                if x >= X % TX and X % TX != 0:
                    loadText += "  ( {} < {} ) ?  {} : VALUE(0.0);\n".format(
                        tileIdx, xthreads - 1, ldgText)
                else:
                    loadText += ldgText + ";\n"
            else:
                ldgText = "__ldg( {0} + ({6} {7} + {4}*gridStride)*{2}  + {1}*{3} + {5})".format(
                    array, x, X, xthreads, u, tileIdx, idx if not noIndex else 0,
                    " / (1024*5000)" if truncatedIndex else "")
                if x == (X // xthreads) and X % xthreads != 0:
                    loadText += "  ( {} < {} ) ?  {} : VALUE(1.0);\n".format(
                        tileIdx, X % xthreads, ldgText)
                elif x > (X // xthreads) or (x == (X // xthreads) and X % xthreads == 0):
                    loadText += "  VALUE(1.0);\n".format(tileIdx, X % TX, ldgText)
                else:
                    loadText += ldgText + ";\n"
            return loadText

            return loadText

        if leapFrog:
            for u in range(0, unroll):
                for m in range(0, TM):
                    self.text += "  {} vANow_{}_{} = VALUE(0.0);\n".format(dtype, m, u)
                self.text += "\n"
                for n in range(0, TN):
                    self.text += "  {} vBNow_{}_{} = VALUE(0.0);\n".format(dtype, n, u)
            # Check if size is even that big
            self.text += "  if( sliceId < K) {\n"
            for u in range(0, unroll):
                for m in range(0, TM):
                    self.text += generateLoad("vANow", "A", "  ", M, TM, m, u, "midx",
                                              mthreads, "sliceId")
                self.text += "\n"
                for n in range(0, TN):
                    self.text += generateLoad("vBNow", "B", "  ", N, TN, n, u, "nidx",
                                              nthreads, "sliceId")
            self.text += "  }\n"

        ## -------------   iteration loop --------
        self.text += "  int64_t idx = sliceId;\n"
        self.text += "  for(idx = sliceId; idx < K{0}; idx += gridStride*{1}){{\n".format(
            " - gridStride" if leapFrog else "", unroll)

        for u in range(0, unroll):
            if leapFrog:
                for m in range(0, TM):
                    self.text += generateLoad("vANext", "A", "    " + dtype, M, TM, m, u,
                                              "midx", mthreads, "idx + gridStride")

                self.text += "\n"
                for n in range(0, TN):
                    self.text += generateLoad("vBNext", "B", "    " + dtype, N, TN, n, u,
                                              "nidx", nthreads, "idx + gridStride")
            else:
                for m in range(0, TM):
                    self.text += generateLoad("vANow", "A", "    " + dtype, M, TM, m, u,
                                              "midx", mthreads, "idx")

                self.text += "\n"
                for n in range(0, TN):
                    self.text += generateLoad("vBNow", "B", "    " + dtype, N, TN, n, u,
                                              "nidx", nthreads, "idx")

            self.text += "\n"
            for m in range(0, TM):
                for n in range(0, TN):
                    self.text += "    tS{3}_{4} = {5}(tS{3}_{4},  vANow_{0}_{2}, vBNow_{1}_{2});\n".format(
                        m, n, u, m if not singleAccumulator else 0,
                        n if not singleAccumulator else n % 2, axpy_func[dtype])

            if leapFrog:
                for m in range(0, TM):
                    self.text += "    vANow_{0}_{1} = vANext_{0}_{1};\n".format(m, u)

                self.text += "\n"
                for n in range(0, TN):
                    self.text += "    vBNow_{0}_{1} = vBNext_{0}_{1};\n".format(n, u)

        self.text += "  }\n"

        ## -------- post iteration ----
        if leapFrog:
            self.text += "  if( idx < K) {\n"
            for u in range(0, unroll):
                for m in range(0, TM):
                    for n in range(0, TN):
                        self.text += "    tS{0}_{1} = {3}(tS{0}_{1}, vANow_{0}_{2}, vBNow_{1}_{2});\n".format(
                            m, n, u, axpy_func[dtype])
            self.text += "  }\n"

        ## ----------- reduction ----------------
        if reduction == "localAtomic":
            self.text += "  int participants =  __syncthreads_count(true);\n"
            #self.text += "if(participants != 256) printf(\" %d participants  \\n \", participants);\n"

        for m in range(0, TM):
            for n in range(0, TN):

                conditionals = []
                if not transposed:
                    if m >= M % TM and M % TM != 0:
                        conditionals.append("{0} < " + str(mthreads - 1))
                    if n >= (N % TN) and N % TN != 0:
                        conditionals.append("{1} < " + str(nthreads - 1))

                else:
                    if M - m * mthreads < 1:
                        conditionals.append("false")
                    if N - n * nthreads < 1:
                        conditionals.append("false")

                    if M - m * mthreads < mthreads:
                        conditionals.append("{0} < " + str(M - m * mthreads))
                    if N - n * nthreads < nthreads:
                        conditionals.append("{1} < " + str(N - n * nthreads))

                writeProtectText = ""
                if len(conditionals) > 0:
                    writeProtectText += "  if ( " + " && ".join(conditionals) + "  )\n  "

                redText = ""
                if reduction == "globalAtomic":
                    self.text += writeProtectText.format("midx", "nidx")
                    if transposed:
                        self.text += (
                            "  atomicAdd(C + ({2}*{0} + midx ) * {4} +  ({3}*{1} + nidx), "
                            "tS{2}_{3});\n").format(mthreads, nthreads, m, n, N)
                    else:
                        self.text += (
                            "  atomicAdd(C + ({2} + midx*{0} ) * {4} +  ({3} + nidx*{1}), "
                            "tS{2}_{3});\n").format(TM, TN, m, n, N)
                elif reduction == "localAtomic":
                    self.text += "  __syncthreads();\n"
                    self.text += "  atomicAdd(&blockResults[midx][nidx], tS{0}_{1});\n".format(
                        m, n)
                    self.text += "  __syncthreads();\n"
                    self.text += "  for(int id = threadIdx.x; id < {}; id += participants) {{\n".format(
                        mthreads * nthreads)
                    self.text += "    int tMidx = id % {};\n".format(mthreads)
                    self.text += "    int tNidx = id / {};\n".format(mthreads)
                    self.text += writeProtectText.format("tMidx", "tNidx")
                    self.text += ("    atomicAdd(C + (tMidx + {0})*{2} + tNidx + {1}, "
                                  "blockResults[tMidx][tNidx]);\n").format(m, n, N)
                    self.text += "    blockResults[tMidx][tNidx] = VALUE(0.0);\n"
                    self.text += "  }\n"
                elif reduction == "store":
                    self.text += writeProtectText.format("midx", "nidx")
                    self.text += ("      C[(midx*{0} + {2} ) * {4} +  (nidx*{1} + {3})]"
                                  "= tS{2}_{3};\n").format(TM, TN, m, n, N)
                else:
                    self.text += "  if(tidx == 123123) \n"
                    self.text += ("      C[(midx*{0} + {2} ) * {4} +  (nidx*{1} + {3})]"
                                  "= tS{2}_{3};\n").format(TM, TN, m, n, N)

        self.text += "} \n"

        #print(self.text)

    def genAddresses(self, tidx):
        addresses = []

        mthreads = (self.M - 1) // self.TM + 1
        nthreads = (self.N - 1) // self.TN + 1

        for m in range(0, self.TM):
            addresses.append(("A", (tidx // mthreads // nthreads * self.M +
                                    (tidx % mthreads) * self.TM + m) * 8))
        for n in range(0, self.TN):
            addresses.append(("B", (tidx // mthreads // nthreads * self.N +
                                    ((tidx // mthreads) % nthreads) * self.TN + n) * 8))

        return addresses

    def run(self, A, B, C, K, blocksPerMP=-1):

        if self.multiprocessor_count is None:
            self.multiprocessor_count = drv.Context.get_device().get_attributes()[
                drv.device_attribute.MULTIPROCESSOR_COUNT]
        if self.mod is None:
            self.mod = SourceModule(self.text, arch="sm_70", options=["-lineinfo"])
        if self.function is None:
            self.function = self.mod.get_function(self.name)
            self.function.prepare(('P', 'P', 'P', numpy.int64))

        minimumBlockCount = max(
            1, (self.M * self.N // self.TM // self.TN - 1) // self.blockSize + 1)

        if self.blockSize * self.function.num_regs > pycuda.tools.DeviceData().registers:
            return

        tb_per_mp = self.estimateBlockCount(self.function.num_regs)

        threadsPerSlice = math.ceil(self.M / self.TM) * math.ceil(self.N / self.TN)

        block = (self.blockSize, 1, 1)
        if blocksPerMP == -1:
            grid = (self.multiprocessor_count * tb_per_mp,)
        else:
            grid = (80,)


        #self.function.set_cache_config(drv.shared_config.PREFER_EQUAL)

        self.function.prepared_call(grid, block, A, B, C, numpy.int64(K))

    def estimateBlockCount(self, registers):
        return min(64 // (self.blockSize // 32),
                   65536 // (max(32, registers) * self.blockSize))
