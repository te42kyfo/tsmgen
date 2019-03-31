
import pycuda.driver as drv


maxBufferElements = 512 * 1024 * 1024

A_gpu = drv.mem_alloc(maxBufferElements * 8)
B_gpu = drv.mem_alloc(maxBufferElements * 8)
C_gpu = drv.mem_alloc(128 * 128 * 8)


def printSASS(code):
    cubin = pycuda.compiler.compile(code, options=["-w", "-std=c++11"], arch="sm_70")

    run(["echo \"" + code + "\" >> temp.cubin"], stdout=PIPE, shell=True)

    newFile = open("temp.cusbin", "wb")
    newFile.write(cubin)
    newFile.close()

    result = run(["nvdisasm -c   temp.cusbin"], stdout=PIPE, shell=True)

    print(len(result.stdout.decode("utf-8").split("\n")))

    print(result.stdout.decode("utf-8"))

    newFile = open("temp.disasm", "wb")
    newFile.write(result.stdout)
    newFile.close()


def benchKernel(kernel, K, iters=10):
    def timeKernel():
        start = drv.Event()
        end = drv.Event()

        start.record()
        kernel.run(A_gpu, B_gpu, C_gpu, K)
        end.record()
        end.synchronize()
        return end.time_since(start)

    dts = [timeKernel() for i in range(0, iters)]

    dts.sort()
    dt = dts[1 if len(dts) > 2 else 0]

    bw = (kernel.M * K + kernel.N * K) * 8 / dt / 10**6
    flops = kernel.M * K * kernel.N * 2 / dt / 10**6

    print("{:5.2f} {:6.0f} {:6.0f}".format(dt, bw, flops))
    return flops
