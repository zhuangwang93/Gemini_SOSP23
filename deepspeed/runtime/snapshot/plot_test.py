import numpy as np
import matplotlib.pyplot as plt


def comm_time(size):
    gpus = 8
    bandwidth = 400 / 8 * 1000
    time = size / (bandwidth / gpus) * 1000
    return time


sizes = [2**i for i in range(4, 10)]
print(sizes)
times = [comm_time(size) for size in sizes]
print(times)

plt.plot(sizes, times, marker="s")
plt.xlabel('Block size (MB)')
plt.ylabel('Time (ms)')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
plt.savefig(f'block_time.png')