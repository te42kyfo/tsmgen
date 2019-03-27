def predictL1(kernel):
    warpAddresses = [kernel.genAddresses(warpLane) for warpLane in range(0, 32)]

    L1CLs = [
        set([warpAddresses[warpLane][load][1] // 128 for warpLane in range(0, 32)])
        for load in range(len(warpAddresses[0]))
    ]

    estRegCount = (kernel.TM * kernel.TN + kernel.TM + kernel.TN) * 2
    if estRegCount > 255:
        for i in range(255, estRegCount):
            L1CLs.append({0})

    L1cycles = sum([max(2, len(load)) for load in L1CLs])

    return kernel.TM * kernel.TN / max(L1cycles, kernel.TM * kernel.TN) * 1.38 * 2 * 32 * 80


def predictMemory(kernel):
    addresses = [[(l[0], l[1] // 64) for l in kernel.genAddresses(warpLane)]
                 for warpLane in range(0, kernel.blockSize)]

    CLs = set([
        addresses[warpLane][load] for warpLane in range(0, kernel.blockSize)
        for load in range(len(addresses[0]))
    ])

    estRegCount = (kernel.TM * kernel.TN + kernel.TM + kernel.TN) * 2 + 10
    estBlockCount = kernel.estimateBlockCount(min(255, estRegCount))
    spillToMemory = max(0, estRegCount - 255 - 24)

    for i in range(0, spillToMemory):
        CLs = CLs.union({("LDL", i)})
        CLs = CLs.union({("STL", i)})

    aluInstrPerLoop = 10
    instrExecLatency = max(aluInstrPerLoop * 2,
                           aluInstrPerLoop * estBlockCount * (kernel.blockSize / 32) / 2)

    latencyPred = 80 * estBlockCount * kernel.TM * kernel.TN * kernel.blockSize * 2 * (
        1.38 / (436 + kernel.TN * kernel.TM * 4 +
                (kernel.TM + kernel.TN) * 2 + instrExecLatency + spillToMemory / 4 * 436))

    throughputPred = 2 * kernel.TM * kernel.TN * kernel.blockSize / (len(CLs) * 64 / 880e9) / 1e9
    return (latencyPred, throughputPred)


def predictALU(kernel):
    aluInstrPerLoop = 10
    GIts = 1.38 * 80 * 2 * 32 / aluInstrPerLoop
    return GIts * kernel.TM * kernel.TN
