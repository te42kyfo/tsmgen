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

revision = 1

for MN in range(1, 64, 1):
    print(MN)
    for TN in range(1, MN + 1):
        if TN > MN / 2 and MN % TN != 0:
            continue
        print("{:2}".format(TN))
        for blockSize in [128, 256, 512]:
            unroll = 1
            print("{:5}".format(blockSize), end="   ")
            kernel = TSMMKernel(MN, MN, TN, blockSize)
            KK = maxBufferElements // min(kernel.M, kernel.N)

            time, flops, bw = benchKernel(kernel, KK, 10)
            clock, power, temp = 0, 0, 0

            print("{:5.2f} {:6.0f} {:6.0f} ".format(time, bw, flops))
