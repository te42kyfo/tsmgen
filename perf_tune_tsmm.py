#!/bin/python

from tsmmkernel import TSMMKernel
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
    print("MN={}".format(MN))
    for TN in [MN, 1, 2, 4, 8, 16, 32]:
        if TN > MN or MN / TN > 16:
            continue
        print(" TN={:2}".format(TN))
        for unroll in [1, 2, 3, 4, 5]:
            for blockSize in [128, 256, 512, 1024]:
                print(" {:2}x   {:5}".format(unroll, blockSize), end="   ")
                kernel = TSMMKernel(MN,
                                    MN,
                                    TN,
                                    blockSize,
                                    unroll,
                                    CSHARED=True,
                                    USETHREADCOUNT=True,
                                    nthreads=TN)
                KK = maxBufferElements // min(kernel.M, kernel.N)

                time, flops, bw = benchKernel(kernel, KK, 10)
                clock, power, temp = 0, 0, 0

                print("{:5.2f} {:6.0f} {:6.0f} ".format(time, bw, flops))
                query = ("DELETE FROM tsmm WHERE M={} AND N={} AND TN={} "
                         "AND blockSize={} AND revision={} AND unroll={} ").format(
                             MN, MN, TN, blockSize, revision, unroll)

                conn.execute(query)
                query = (
                    "INSERT INTO tsmm"
                    "(revision, M, N, TN, blockSize, unroll, K, time, bw, flops) VALUES"
                    "({}, {}, {}, {}, {}, {}, {}, {}, {}, {})").format(
                        revision, MN, MN, TN, blockSize, unroll, KK, time, bw, flops)
                conn.execute(query)
        conn.commit()

conn.close()
