import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('benchmarks_new3.db')
c = conn.cursor()

fig, ax = plt.subplots()
fig.set_figwidth(6)
fig.set_figheight(4)

ax.plot(np.arange(1, 65),
        np.minimum(7066,
                   np.arange(1, 65) / 8 * 860),
        "--",
        color="gray",
        label="memory bandwidth limit")
ax.plot(np.arange(1, 65),
        np.ones((64)) * 7066,
        ":",
        color="gray",
        label="FP execution limit")

rev = 5

for rev in [4, 5, 6]:
    bests = []
    msizes = []
    nsizes = []
    for MN in range(1, 65):
        c.execute("SELECT * from tsmttsm WHERE M={0} AND N={0} and datatype=\"D\" and revision={1}".format(
            MN, rev))
        results = list(c)

        if len(results) > 0:
            results.sort(key=lambda r: r[14])
            bests.append((results[-1][2], results[-1][14]))
            msizes.append((results[-1][3], results[-1][4]))
            nsizes.append((results[-1][2], results[-1][3] * results[-1][4]))

    print(str(rev) + ": " + str(len(bests)))
    print(bests)
    if (len(bests) > 0):
        ax.plot(*zip(*bests), "o-", label="Tiles", markersize=3)

conn.close()

data_cublas = np.asarray([
    5, 6, 10, 11, 16, 21, 27, 23, 26, 29, 34, 39, 45, 51, 58, 65, 52, 42, 45, 48, 50, 52,
    55, 59, 60, 62, 65, 68, 63, 65, 68, 72, 98, 96, 97, 102, 103, 107, 109, 114, 114, 118,
    119, 126, 126, 129, 130, 138, 135, 137, 139, 144, 145, 149, 150, 154, 152, 156, 156,
    162, 162, 162, 166, 171
]) / 8 * np.arange(1, 65)

data_cutlas = np.asarray([
    0.647439, 2.59042, 5.82902, 10.356, 16.1832, 23.304, 31.7123, 41.3783, 52.3969,
    64.6835, 78.226, 93.1248, 109.275, 126.749, 145.448, 165.535, 186.754, 209.397,
    216.252, 258.441, 284.819, 312.531, 341.567, 371.84, 403.28, 436.06, 470.002, 505.396,
    542.228, 579.956, 619.421, 659.329, 701.089, 743.79, 787.811, 832.718, 879.394,
    926.741, 975.308, 1025.74, 1076.11, 1127.62, 1180.71, 1235.8, 1290.55, 1346.79,
    1403.34, 1465.28, 1520.92, 1581.48, 1642.36, 1704.54, 1767.55, 1832.28, 1895.25,
    1962.45, 2025.63, 2091.06, 2156.29, 2223.87, 2292.32, 2360.89, 2429.82
])


tsmm_cublas = [
    7, 79, 178, 317, 451, 576, 637, 783, 758, 855, 916, 1198, 1099, 1224, 1255, 1633,
    1302, 1423, 1475, 1602, 1659, 1747, 1786, 2114, 2059, 2192, 2189, 2555, 2335, 2533,
    2479, 3081, 2024, 2136, 2182, 2408, 2347, 2471, 2502, 2718, 2926, 3078, 3162, 3398,
    3341, 3510, 3489, 3899, 3527, 3707, 3697, 4048, 3891, 4027, 4029, 4601, 4079, 4234,
    4218, 4586, 4381, 4590, 4611, 5275
]

#ax.plot(data_cublas, "o-", label="CUBLAS", markersize=3)
#ax.plot(data_cutlas, "o-", label="CUTLASS", markersize=3)
ax.plot(tsmm_cublas, "o-", label="CUBLAS", markersize=3)

ax.grid(True)
ax.set_ylim((0, ax.get_ylim()[1]))
ax.set_xticks([1] + list(range(8, 64 + 1, 8)))
ax.set_xlabel("M=N")
ax.set_ylabel("GFlop/s")
ax.legend(loc=2)
fig.savefig("best_perf_presentation.png", dpi=300, pad_inches=0.0, bbox_inches="tight")

#ax2.set_yticks([1] + list(range(5, 110, 5)))
#ax2.set_xticks([1] + list(range(2, 64, 1)))

plt.show()


