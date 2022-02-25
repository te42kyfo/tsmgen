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

conn = sqlite3.connect('benchmarks.db')
c = conn.cursor()

revision = 7

for MN in range(1, 65, 1):
    print("MN={}".format(MN))
    for dtype in ["double", "cuDoubleComplex"]:

        for options in [{
                "CVALS": False,
                "CSHARED": False
        }, {
                "CVALS": True,
                "CSHARED": False
        }, {
                "CVALS": False,
                "CSHARED": True
        }]:
            print(str(options))
            for TN in [MN, 1, 2, 4, 8, 16, 32]:
                if TN > MN or MN / TN > 16:
                    continue
                print(" TN={:2}".format(TN))
                for unroll in [1, 2, 3, 4]:
                    for blockSize in [128, 256, 512, 1024]:

                        print("{} {:4} {:2} {:2}x   {:5}".format(
                            "D" if dtype == "double" else "Z", MN, TN, unroll, blockSize),
                              end="   ")

                        kernel = TSMMKernel(MN,
                                            MN,
                                            TN,
                                            blockSize,
                                            unroll,
                                            USETHREADCOUNT=True,
                                            dtype=dtype,
                                            **options,
                                            nthreads=TN)

                        KK = maxBufferElements // min(kernel.M, kernel.N) / 2

                        #try:
                        time, flops, bw = benchKernel(kernel, KK, 10)

                        #except (KeyboardInterrupt, SystemExit):
                        #    raise
                        #except:
                        #    print()
                        #    continue
                        clock, power, temp = 0, 0, 0

                        print("{:5.2f} {:6.0f} {:6.0f} ".format(time, bw, flops))
                        query = ("DELETE FROM tsmm WHERE M={} AND N={} AND TN={} "
                                 "AND blockSize={} AND revision={} AND unroll={} "
                                 "AND datatype=\"{}\" AND options=\"{}\"").format(
                                     MN, MN, TN, blockSize, revision, unroll,
                                     ("D" if dtype == "double" else "Z"), str(options))

                        conn.execute(query)
                        query = (
                            "INSERT INTO tsmm"
                            "(revision, M, N, datatype, TN, blockSize, options, unroll, "
                            "K, time, bw, flops) VALUES"
                            "({}, {}, {}, \"{}\", {},  {}, \"{}\","
                            "{}, {}, {}, {}, {})").format(
                                revision, MN, MN, ("D" if dtype == "double" else "Z"), TN,
                                blockSize, str(options), unroll, KK, time, bw, flops)
                        conn.execute(query)

                conn.commit()

conn.close()
