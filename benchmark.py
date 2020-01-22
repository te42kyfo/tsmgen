from pynvml import *
import pycuda.driver as drv
import pycuda
import sys
from subprocess import run, PIPE

nvmlInit()

maxBufferElements = 512 * 1024 * 1024

A_gpu = drv.mem_alloc(maxBufferElements * 8)
B_gpu = drv.mem_alloc(maxBufferElements * 8)
C_gpu = drv.mem_alloc(128 * 128 * 8 * 2)


def printSASS(code):
    cubin = pycuda.compiler.compile(code, options=["-w", "-std=c++11"], arch="sm_70")

    run(["echo \"" + code + "\" >> temp.cubin"], stdout=PIPE, shell=True)

    newFile = open("temp.cusbin", "wb")
    newFile.write(cubin)
    newFile.close()

    result = run(["nvdisasm  temp.cusbin"], stdout=PIPE, shell=True)

    print(len(result.stdout.decode("utf-8").split("\n")))

    print(result.stdout.decode("utf-8"))

    newFile = open("temp.disasm", "wb")
    newFile.write(result.stdout)
    newFile.close()


def measurePower(kernel, K):
    device = nvmlDeviceGetHandleByIndex(0)
    start = drv.Event()
    end = drv.Event()

    start.record()
    kernel.run(A_gpu, B_gpu, C_gpu, K)
    kernel.run(A_gpu, B_gpu, C_gpu, K)
    temp = nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU)
    clock = nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM)
    power = nvmlDeviceGetPowerUsage(device)

    end.record()
    end.synchronize()
    return (clock, power, temp)


def benchKernel(kernel, K, iters=10, blocksPerMP=-1):

    kernel.run(A_gpu, B_gpu, C_gpu, K, blocksPerMP)

    #if kernel.function.num_regs == 255:
    #    return (1, 1, 1)

    def timeKernel():
        start = drv.Event()
        end = drv.Event()

        start.record()
        kernel.run(A_gpu, B_gpu, C_gpu, K, blocksPerMP)

        end.record()
        end.synchronize()
        return end.time_since(start)

    dts = [timeKernel() for i in range(0, iters)]

    dts.sort()
    dt = dts[1 if len(dts) > 2 else 0]

    bw = (kernel.M * K +
          kernel.N * K) * (2 if kernel.dtype == "cuDoubleComplex" else 1) * 8 / dt / 10**6
    flops = kernel.M * K * kernel.N * (8 if kernel.dtype == "cuDoubleComplex" else
                                       2) / dt / 10**6

    return (dt, flops, bw)
