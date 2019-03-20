from kernel import Kernel
import numpy as np
import pycuda.driver as drv


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
    if np.sum(np_ref - C) != 0:
        print("  -- Verification Fail, wrong Results --")
        print(K)
        print(np_ref)
        print(C)
    print(str(K) + " ", end="", flush=True)


def testSeries(kernel):
    for i in range(0, 10):
        krange = 10**np.random.randint(1, 6)
        testKernel(kernel, np.random.randint(1, krange))
    print()


for i in range(1, 33):
    for t in range(1, 17):
        if i % t != 0:
            continue
        print(str(t) + " " + str(i) + ":  ")
        kernel = Kernel(i, i, t, t, 256)
        testSeries(kernel)
