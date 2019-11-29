#!/bin/python

from kernel import Kernel
from predict import *
from benchmark import *
import numpy as np
import pycuda.driver as drv
import pycuda

import sys
from subprocess import run, PIPE

import matplotlib
matplotlib.use("SVG")
import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('benchmarks_new.db')
c = conn.cursor()

revision = 6

for MN in range(1, 65, 1):
    print(MN)
    for TM in range(1, MN + 1):
        for TN in range(1, MN + 1):
            if (MN % TM != 0 and TM * 2 > MN + 1) or (MN % TN != 0 and TN * 2 > MN + 1):
                continue

            print("{:2} {:2}".format(TM, TN))
            for reductionType in ["localAtomic"]:
                for leapFrog in [True, False]:
                    for blockSize in [128, 256]:
                        unroll = 1
                        if blockSize * (TM * TN + TM + TN +
                                        (TM + TN if leapFrog else 0)) * 2 > 65536 or (TM * TN + TM +
                                                                                      TN) * 2 > 255:
                            continue
                        print("{:5}".format(blockSize), end="   ")
                        kernel = Kernel(MN,
                                        MN,
                                        TM,
                                        TN,
                                        blockSize,
                                        reduction=reductionType,
                                        unroll=unroll,
                                        leapFrog=leapFrog)
                        KK = maxBufferElements // min(kernel.M, kernel.N)

                        time, flops, bw = benchKernel(kernel, KK, 10)
                        clock, power, temp = 0, 0, 0
                        #clock, power, temp = measurePower(kernel, KK)

                        print("{:5.2f} {:6.0f} {:6.0f} ".format(time, bw, flops))

                        #c.execute(("SELECT * FROM tsmttsm WHERE "
                        #           "M={} AND N={} AND TN={} AND TM={} AND blockSize={} "
                        #           "AND reduction=\"{}\" and revision={}").format(
                        #               MN, MN, TM, TN, blockSize, "localAtomic", revision - 1))
                        clist = list(c)
                        if (len(list(clist)) > 0):
                            previous = clist[0]
                            print("        {:5.2f}  {:5.0f}  {:5.0f}".format(
                                previous[7], previous[9], previous[8]))
                            print("        {:5.2f}  {:5.0f}  {:5.0f}".format(
                                previous[7] - time, previous[9] - bw, previous[8] - flops))

                        query = (
                            "DELETE FROM tsmttsm WHERE "
                            "M={} AND N={} AND TM={} AND TN={} AND blockSize={} "
                            "AND reduction=\"{}\" AND revision={} AND unroll={} AND leapfrog={}"
                        ).format(MN, MN, TM, TN, blockSize, reductionType, revision, unroll,
                                 1 if leapFrog else 0)

                        conn.execute(query)
                        query = ("INSERT INTO tsmttsm VALUES"
                                 "({}, {}, {}, {}, {},{},\"{}\",{},{},{},{},{}, {})").format(
                                     revision, MN, MN, TM, TN, blockSize, reductionType, unroll,
                                     1 if leapFrog else 0, KK, time, bw, flops)
                        conn.execute(query)
                    conn.commit()

conn.close()
