#!/bin/python

import pycuda


def genCode(M, N, TM, TN):
    dtype = "double"
    code = "void __global__ tsmttsm_{0}_{1}_{2}_{3}({4}* A, {4}* B, {4}* C, int K) {{\n".format(M, N, TM, TN, dtype)
    code += "    int tidx = threadIdx.x + blockIdx.x*blockDim.x;\n"

    code += "}\n"
    return code


print(genCode(4, 4, 2, 2))
