import pycuda
import pycuda.autoinit

import pycuda.driver as drv
import numpy

from pycuda.compiler import SourceModule


class Kernel:

    def __init__(self, M, N, TM, TN, blockSize, reduction="globalAtomic"):
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

        #if M % TM != 0 or N % TN != 0:
        #    print("Tilesize does not match: M % TM != 0 or N % TN != 0\n")

        mthreads = (M - 1) // TM + 1
        nthreads = (N - 1) // TN + 1
        threadsPerSlice = mthreads * nthreads

        dtype = "double"
        self.text = "void __global__ __launch_bounds__({2}) {0} ({1}* A, {1}* B, {1}* C, size_t K) {{\n".format(
            self.name, dtype, self.blockSize)
        self.text += "  int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"
        self.text += "  int sliceId = tidx / {};\n".format(threadsPerSlice)
        self.text += "  int midx = tidx % {};\n".format(mthreads)
        self.text += "  int nidx = (tidx / {})  % {};\n".format(mthreads, nthreads)
        self.text += "\n"
        if reduction == "localAtomic":
            self.text += "  __shared__ {} blockResults[{}][{}];\n".format(
                dtype, mthreads, nthreads)
            self.text += "for( int id = threadIdx.x; id <  {}; id += {})\n".format(mthreads*nthreads, self.blockSize)
            self.text += "    blockResults[id/{0}][id%{0}] = 0.0;\n".format(mthreads)
 
            self.text += "     __syncthreads();\n"

        self.text += "  if( tidx >= (gridDim.x*blockDim.x/{0})*{0}) return;\n\n".format(
            threadsPerSlice)

        for m in range(0, TM):
            self.text += "  {} ".format(dtype)
            first = True
            for n in range(0, TN):
                if not first:
                    self.text += ", "
                first = False
                self.text += "tS{}_{} = 0".format(m, n)

            self.text += ";\n"
        self.text += "\n"

        self.text += "  for( size_t idx = sliceId; idx < K; idx += (gridDim.x*{})/{}){{\n".format(
            blockSize, threadsPerSlice)

        for m in range(0, TM):
            self.text += "    {0} vA_{1} = __ldg( A + idx*{2}  + midx*{3} + {1});\n".format(
                dtype, m, M, TM)

        for n in range(0, TN):
            self.text += "    {0} vB_{1} = __ldg( B + idx*{2}  + nidx*{3} + {1});\n".format(
                dtype, n, N, TN)

        for m in range(0, TM):
            for n in range(0, TN):
                if (M % TM == 0 and N % TN == 0) or (m < M % TM and n < N % TN):
                    self.text += "    tS{0}_{1} += vA_{0} * vB_{1};\n".format(m, n)

        self.text += "  }\n"

        if reduction == "localAtomic":
            self.text += "  int participants =  __syncthreads_count(true);\n"
            #self.text += "if(participants != 256) printf(\" %d participants  \\n \", participants);\n"

        for m in range(0, TM):
            for n in range(0, TN):

                if reduction == "globalAtomic":
                    self.text += (
                        "  atomicAdd(C + (midx*{0} + {2} ) * {4} +  (nidx*{1} + {3}), "
                        "tS{2}_{3});\n").format(TM, TN, m, n, N)
                elif reduction == "localAtomic":
                    self.text += "  __syncthreads();\n"
                    self.text += "  atomicAdd(&blockResults[midx][nidx], tS{0}_{1});\n".format(
                        m, n)
                    self.text += "  __syncthreads();\n"
                    self.text += "  for(int id = threadIdx.x; id < {}; id += participants) {{\n".format(
                        mthreads * nthreads)
                    self.text += "    int tMidx = id / {};\n".format(mthreads)
                    self.text += "    int tNidx = id % {};\n".format(nthreads)

                    self.text += ("    atomicAdd(C + (tMidx + {0})*{2} + tNidx + {1}, "
                                  "blockResults[tMidx][tNidx]);\n").format(m, n, N)
                    self.text += "    blockResults[tMidx][tNidx] = 0.0;\n"
                    self.text += "  }\n"
                elif reduction == "store":
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
            addresses.append(("B", (tidx // mthreads // nthreads * self.N + (
                (tidx // mthreads) % nthreads) * self.TN + n) * 8))

        return addresses

    def run(self, A, B, C, K):

        if self.multiprocessor_count is None:
            self.multiprocessor_count = drv.Context.get_device().get_attributes()[
                drv.device_attribute.MULTIPROCESSOR_COUNT]
        if self.mod is None:
            self.mod = SourceModule(self.text, arch="sm_70")
        if self.function is None:
            self.function = self.mod.get_function(self.name)
            self.function.prepare(('P', 'P', 'P', numpy.int64))

        minimumBlockCount = max(
            1, (self.M * self.N // self.TM // self.TN - 1) // self.blockSize + 1)

        if self.blockSize * self.function.num_regs > pycuda.tools.DeviceData().registers:
            return

        tb_per_mp = self.estimateBlockCount(self.function.num_regs)

        block = (self.blockSize, 1, 1)
        grid = (max(self.multiprocessor_count * tb_per_mp * 2, minimumBlockCount),)

        self.function.prepared_call(grid, block, A, B, C, numpy.int64(K))

    def estimateBlockCount(self, registers):
        return min(64 // (self.blockSize // 32),
                   65536 // (max(32, registers) * self.blockSize))
