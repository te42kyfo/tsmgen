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

    print(instrExecLatency)
    latencyPred = 80 * estBlockCount * kernel.TM * kernel.TN * kernel.blockSize * 2 * (
        1.38 / (436 + kernel.TN * kernel.TM * 4 +
                (kernel.TM + kernel.TN) * 2 + instrExecLatency + spillToMemory / 4 * 436))

    throughputPred = 2 * kernel.TM * kernel.TN * kernel.blockSize / (len(CLs) * 64 / 880e9) / 1e9
    return (latencyPred, throughputPred)


def predictALU(kernel):
    aluInstrPerLoop = 10
    GIts = 1.38 * 80 * 2 * 32 / aluInstrPerLoop
    return GIts * kernel.TM * kernel.TN


def predictUniform(kernel):
    alu_instr = 10
    alu_lat = 4
    alu_rcpt = 2

    dp_instr = kernel.TM * kernel.TN
    dp_lat = 4
    dp_rcpt = 4

    warpAddresses = [kernel.genAddresses(warpLane) for warpLane in range(0, 32)]

    L1CLs = [
        set([warpAddresses[warpLane][load][1] // 128 for warpLane in range(0, 32)])
        for load in range(len(warpAddresses[0]))
    ]
    L1cycles = sum([max(2, len(load)) for load in L1CLs])

    print(L1cycles)
    
    memAddresses = [[(l[0], l[1] // 64) for l in kernel.genAddresses(warpLane)]
                    for warpLane in range(0, kernel.blockSize)]
    memCLs = set([
        memAddresses[warpLane][load] for warpLane in range(0, kernel.blockSize)
        for load in range(len(memAddresses[0]))
    ])

    estRegCount = (kernel.TM * kernel.TN + kernel.TM + kernel.TN) * 2 + 10
    estBlockCount = kernel.estimateBlockCount(min(255, estRegCount))

    warps = estBlockCount * kernel.blockSize / 32 / 4


    t_alu = alu_instr * (alu_lat * max(0, warps / (alu_lat / alu_rcpt)))
    t_dp = dp_instr * (dp_lat * max(0, warps / (dp_lat / dp_rcpt)))
    t_l1 = L1cycles * warps
    t_mem = max(436, 64 * len(memCLs) * estBlockCount * 80 / 880 * 1.38)
    t_total = t_alu + t_dp + max(t_l1, t_mem)


    print(t_alu)
    print(t_dp)
    print(t_l1)
    print(t_mem)
    print(t_total)

    #for i in range(0, 10):
    #    t_alu = alu_instr * (alu_lat + max(0, warps * t_alu / t_total - alu_lat / alu_rcpt))
    #    t_dp = dp_instr * (dp_lat + max(0, warps * t_dp / t_total - dp_lat / dp_rcpt))
    #    t_l1 = L1cycles * (1 + max(0, warps * t_l1 / t_total))
    #    t_mem = max(436, 64 * len(memCLs) * estBlockCount * t_mem / t_total * 80 / 880 * 1.38)
    #    t_total = t_alu + t_dp + max(t_l1, t_mem)
    #print(t_alu)
    #print(t_dp)
    #print(t_l1)
    #print(t_mem)
    #print(t_total)

    return 1.38 / t_total * 80 * 4 * 32 * warps * kernel.TM * kernel.TN * 2
