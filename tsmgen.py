#!/bin/python

import pycuda


def genCode(M, N, TM, TN, blockSize):
    if M % TM != 0 or N % TN != 0:
        print("Tilesize does not match: M % TM != 0 or N % TN != 0\n")
        return ""

    mthreads = M / TM
    nthreads = N / TN
    threadsPerTile = TM * TN

    dtype = "double"
    code = "void __global__ tsmttsm_{0}_{1}_{2}_{3}({4}* A, {4}* B, {4}* C, int K) {{\n".format(M, N, TM, TN, dtype)
    code += "    int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"
    code += "    int sliceId = tidx / {};\n".format(threadsPerTile)
    code += "    int midx = tidx % {};\n".format(mthreads)
    code += "    int nidx = (tidx / {})  % {};\n".format(mthreads, nthreads)
    code += "\n"

    code += "    if( tidx >= (tidx/{0})*{0}) return;\n\n".format(threadsPerTile)

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

    code += "    for( int idx = sliceId; idx < K; idx += (gridDim.x*{})/{}){{\n".format(blockSize, threadsPerTile)

    for m in range(0, TM):
        for n in range(0, TN):
            code += "        tS{0}_{1} += A[idx*{2} + midx*{4} + {0}] * B[idx*{3} + nidx*{5} + {1}];\n".format(
                m, n, M, N, TM, TN)

    code += "    }\n"


    for m in range(0, TM):
        for n in range(0, TN):
            code += "    atomicAdd(C + (midx*{0} + {2}) * N +  (nidx*{1} + {3}), tS{2}_{3});\n".format(TM, TN, m, n)

    code += "}\n"
    return code


print(genCode(4, 4, 2, 2, 256))

print("---")

print(genCode(20, 8, 10, 1, 256))
