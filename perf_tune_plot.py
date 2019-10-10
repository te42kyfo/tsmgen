
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import sqlite3

conn = sqlite3.connect('benchmarks.db')
c = conn.cursor()

fig, ax = plt.subplots()
fig.set_figwidth(6)
fig.set_figheight(4)

fig2, ax2 = plt.subplots()
fig2.set_figwidth(6)
fig2.set_figheight(4)


ax.plot(np.arange(1,65), np.minimum( 7066, np.arange(1,65) / 8 * 860), "--", color="gray", label="memory bandwidth limit")
ax.plot(np.arange(1,65), np.ones((64))*7066, ":", color="gray", label="FP execution limit")


bests = []
msizes = []
nsizes = []
rev = 4
for MN in range(1, 65):
    c.execute("SELECT * from tsmttsm WHERE M={0} AND N={0}  and revision={1}".format(MN, rev))
    results = list(c)

    if len(results) > 0:
        results.sort(key=lambda r: r[10])
        bests.append((results[-1][2], results[-1][10]))
        msizes.append((results[-1][3], results[-1][4]))
        nsizes.append((results[-1][2], results[-1][3] * results[-1][4]))

        #sizes.append(
        #    (results[-1][2], results[-1][3] * results[-1][4] / (results[-1][4] + results[-1][3])))

print(str(rev) + ": " + str(len(bests)))
ax.plot(*zip(*bests), "o-", label="Tiles", markersize=3)
ax2.plot(*zip(*msizes), "*", color="C" + str(rev))
#ax2.plot(*zip(*msizes), "--*",color="C" +str(rev))

bests = []
for MN in range(1, 65):
    c.execute("SELECT * from tsmttsm WHERE M={0} AND N={0} and TM=1 and TN=1 and revision={1}".format(MN, rev))
    results = list(c)

    if len(results) > 0:
        results.sort(key=lambda r: r[10])
        bests.append((results[-1][2], results[-1][10]))

ax.plot(*zip(*bests), "o-", label="K,M,N", markersize=3)


conn.close()


data_cublas = np.asarray([
    5, 6, 10, 11, 16, 21, 27, 23, 26, 29, 34, 39, 45, 51, 58, 65, 52, 42, 45, 48, 50, 52, 55, 59,
    60, 62, 65, 68, 63, 65, 68, 72, 98, 96, 97, 102, 103, 107, 109, 114, 114, 118, 119, 126, 126,
    129, 130, 138, 135, 137, 139, 144, 145, 149, 150, 154, 152, 156, 156, 162, 162, 162, 166, 171
]) / 8 * np.arange(1, 65)

data_cutlas = np.asarray([
    0.647439, 2.59042, 5.82902, 10.356, 16.1832, 23.304, 31.7123, 41.3783, 52.3969, 64.6835, 78.226,
    93.1248, 109.275, 126.749, 145.448, 165.535, 186.754, 209.397, 216.252, 258.441, 284.819,
    312.531, 341.567, 371.84, 403.28, 436.06, 470.002, 505.396, 542.228, 579.956, 619.421, 659.329,
    701.089, 743.79, 787.811, 832.718, 879.394, 926.741, 975.308, 1025.74, 1076.11, 1127.62,
    1180.71, 1235.8, 1290.55, 1346.79, 1403.34, 1465.28, 1520.92, 1581.48, 1642.36, 1704.54,
    1767.55, 1832.28, 1895.25, 1962.45, 2025.63, 2091.06, 2156.29, 2223.87, 2292.32, 2360.89,
    2429.82
])


ax.plot(data_cublas, "o-", label="CUBLAS", markersize=3)
#ax.plot(data_cutlas, "o-", label="CUTLASS", markersize=3)


ax.grid(True)
ax.set_ylim((0, ax.get_ylim()[1]))
ax.set_xticks([1] + list(range(8, 64 + 1, 8)))
ax.set_xlabel("M=N")
ax.set_ylabel("GFlop/s")
ax.legend(loc=2)
fig.savefig("best_perf_presentation.png", dpi=300, pad_inches=0.0, bbox_inches="tight")



ax2.plot(range(1, 32), 100 / np.arange(1,32))

ax2.yaxis.grid(True)
ax2.set_xlim((0, 24))
ax2.set_ylim((0, 24))
fig2.legend()
#ax2.set_yticks([1] + list(range(5, 110, 5)))
#ax2.set_xticks([1] + list(range(2, 64, 1)))
fig2.savefig("best_perf_size_presentation.png", dpi=300, pad_inches=0.0, bbox_inches="tight")

plt.show()
