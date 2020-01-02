from kernel import Kernel
from tsmmkernel import TSMMKernel
import numpy as np
import pycuda.driver as drv
import termcolor as tc

def testKernel(kernel, K):

    A = np.around(np.random.randn(K, kernel.M).astype(np.float64))
    B = np.around(np.random.randn(K, kernel.N).astype(np.float64))
    C = np.zeros((kernel.M, kernel.N), dtype=np.float64)

    A_gpu = drv.mem_alloc(A.nbytes)
    B_gpu = drv.mem_alloc(B.nbytes)
    C_gpu = drv.mem_alloc(C.nbytes)

    drv.memcpy_htod(A_gpu, A)
    drv.memcpy_htod(B_gpu, B)
    drv.memcpy_htod(C_gpu, C)

    kernel.run(A_gpu, B_gpu, C_gpu, K)

    drv.memcpy_dtoh(C, C_gpu)

    np_ref = np.matmul(np.transpose(A), B)
    passed = True
    if np.sum(np_ref - C) != 0:
        print(tc.colored("  -- Verification Fail, wrong Results --", "red"))
        print(K)
        print(np_ref - C)
        print(C)
        passed = False
    print(str(K) + " ", end="", flush=True)
    return passed

def testTSMMKernel(kernel, K):
    # B = AC
    A = np.around(np.random.randn(K, kernel.M).astype(np.float64))
    B = np.around(np.random.randn(K, kernel.N).astype(np.float64))
    C = np.around(np.random.randn(kernel.M, kernel.N).astype(np.float64))

    A_gpu = drv.mem_alloc(A.nbytes)
    B_gpu = drv.mem_alloc(B.nbytes)
    C_gpu = drv.mem_alloc(C.nbytes)

    drv.memcpy_htod(A_gpu, A)
    drv.memcpy_htod(B_gpu, B)
    drv.memcpy_htod(C_gpu, C)

    kernel.run(A_gpu, B_gpu, C_gpu, K)

    drv.memcpy_dtoh(B, B_gpu)

    np_ref = np.matmul(A, C)
    passed = True
    if np.sum(np_ref - B) != 0:
        print(tc.colored("  -- Verification Fail, wrong Results --", "red"))
        print(K)
        print(np_ref - B)
        print(B)
        passed = False
    print(str(K) + " ", end="", flush=True)
    return passed


def testSeries(kernel):
    passed = True
    for i in range(0, 10):
        krange = 10**np.random.randint(1, 7)
        passed &= testKernel(kernel, np.random.randint(1, krange))
    print()
    return passed


def testTSMMSeries(kernel):
    passed = True
    for i in range(0, 10):
        krange = 10**np.random.randint(1, 7)
        passed &= testTSMMKernel(kernel,  np.random.randint(1, krange))
    print()
    return passed


for m in range(1,32):
    for n in range(m,m+1):
        for tn in range(1, n+1):
            if tn > n/2 and n %tn != 0:
                continue
            print(str(m) + " " + str(n) + " "  + str(tn) + ":  ")
            kernel = TSMMKernel(m, n, tn, 256, 4)

            if not testTSMMSeries(kernel):
                print(kernel.text)
                exit()
