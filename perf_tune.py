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

conn = sqlite3.connect('benchmarks.db')
c = conn.cursor()

revision = 2

for MN in range(1, 65):
    print(MN)
    for TM in range(1, 65):
        for TN in range(1, 65):
            if MN % TN != 0 or MN % TM != 0:
                continue
            if TM * TN > 128:
                continue
            print("{:2} {:2}".format(TM, TN))
            for blockSize in range(128, 1024 + 1, 128):
                for unroll in [1, 2, 4]:
                    if blockSize * TM * TN * 2 > 65536:
                        continue
                    print("{:5}".format(blockSize), end="   ")
                    kernel = Kernel(
                        MN, MN, TM, TN, blockSize, reduction="localAtomic", unroll=unroll)
                    #print(kernel.text)
                    KK = maxBufferElements // min(kernel.M, kernel.N)
                    time, flops, bw = benchKernel(kernel, KK, 10)
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

                    conn.execute(("DELETE FROM tsmttsm WHERE "
                                  "M={} AND N={} AND TM={} AND TN={} AND blockSize={} "
                                  "AND reduction=\"{}\" AND revision={} AND unroll={}").format(
                                      MN, MN, TM, TN, blockSize, "localAtomic", revision, unroll))
                    conn.execute(("INSERT INTO tsmttsm VALUES"
                                  "({}, {}, {},{},{},{},{},{},{},{},{}, {})").format(
                                      revision, MN, MN, TM, TN, blockSize, "\"localAtomic\"",
                                      unroll, KK, time, flops, bw))

        conn.commit()
conn.close()
