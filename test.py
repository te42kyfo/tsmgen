from kernel import Kernel
from tsmmkernel import TSMMKernel
import numpy as np
import pycuda.driver as drv
import termcolor as tc


def testKernel(kernel, K):

    if kernel.dtype == "cuDoubleComplex":
        A = np.around(
            np.random.randn(K, kernel.M * 2).astype(np.float64).view(np.complex128))
        B = np.around(
            np.random.randn(K, kernel.N * 2).astype(np.float64).view(np.complex128))
        C = np.zeros((kernel.M, kernel.N), dtype=np.complex128)
    else:
        A = np.around(np.random.randn(K, kernel.M).astype(np.float64))
        B = np.around(np.random.randn(K, kernel.N).astype(np.float64))
        C = np.zeros((kernel.M, kernel.N), dtype=np.float64)

    #A = np.ones((K, kernel.M), dtype=np.complex128)
    #B = np.ones((K, kernel.N), dtype=np.complex128)

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
        print("Difference")
        print(np_ref - C)
        print("solution")
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
    np.set_printoptions(precision=0, threshold=False)
    if np.sum(np_ref - B) != 0:
        print(tc.colored("  -- Verification Fail, wrong Results --", "red"))
        print(K)
        print("Difference:")
        diff = np_ref - B
        linesPrinted = 0
        for y in range(0, diff.shape[0]):
            for x in range(0, diff.shape[1]):
                if diff[y, x] != 0:
                    linesPrinted += 1
                    if linesPrinted < 200:
                        print(str(y), end=":  ")
                        for x in range(0, diff.shape[1]):
                            print(diff[y, x], end="  ")
                        print(" ------ ", end="")
                        for x in range(0, diff.shape[1]):
                            print(B[y, x], end="  ")
                        print("")
                    break

        print("Lines wrong: " + str(linesPrinted))
        print(B[:100, :])
        passed = False
    print(str(K) + " ", end="", flush=True)
    return passed


def testSeries(kernel):
    passed = True
    for i in range(0, 5):
        krange = 10**np.random.randint(1, 7)
        passed &= testKernel(kernel, 10000)#np.random.randint(1, krange))
    print()
    return passed


def testTSMMSeries(kernel):
    passed = True
    for i in range(0, 10):
        krange = 10**np.random.randint(1, 8)
        passed &= testTSMMKernel(kernel, np.random.randint(1, krange))
    print()
    return passed


for MN in range(45, 65):
    for TM in range(14, MN):
        for TN in range(3, 5):
            for reduction in ["globalAtomic", "localAtomic"]:
                for transposed in [False, True]:
                    for leapFrog in [False, True]:
                        for dtype in ["double", "cuDoubleComplex"]:
                            for blockSize in [128, 256]:
                                if blockSize * (TM * TN + TM + TN +
                                                (TM + TN if leapFrog else 0)) * (
                                                    2 if dtype == "double" else 4
                                                ) > 65536 or (TM * TN + TM + TN) * (
                                                    2 if dtype == "double" else 4) > 255:
                                    continue
                                if (MN % TM != 0 and
                                        TM * 2 > MN + 1) or (MN % TN != 0 and
                                                             TN * 2 > MN + 1):
                                    continue
                                print(
                                    str(MN) + " " + str(MN) + " " + str(TM) + " " +
                                    str(TN) + " " + reduction + " " + str(transposed) +
                                    " " + str(leapFrog) + " " + dtype + " " +
                                    str(blockSize) + " :  ")
                                kernel = Kernel(MN,
                                                MN,
                                                TM,
                                                TN,
                                                blockSize,
                                                reduction=reduction,
                                                dtype=dtype,
                                                transposed=transposed,
                                                leapFrog=leapFrog)

                                if not testSeries(kernel):
                                    print(kernel.text)
                                    exit()
