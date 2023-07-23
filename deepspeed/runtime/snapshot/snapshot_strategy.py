import numpy as np
import matplotlib.pyplot as plt
from heapq import nlargest


class SnapshotStrategy():
    def __init__(self, filename, threshold=10, jump_lines=6, alpha=0.8):
        self.filename = filename
        self.threshold = threshold
        self.jump_lines = jump_lines
        self.alpha = alpha
        self.extract_data()


    def extract_data(self):
        self.gaps = {}
        with open(self.filename, "r") as f:
            for line in f:
                # jump the first few lines 
                if self.jump_lines > 0:
                    self.jump_lines -= 1
                    continue

                gaps = line.strip().split(" ")
                for i, gap in enumerate(gaps):
                    if i not in self.gaps:
                        self.gaps[i] = []
                    if len(gap) > 0:
                        self.gaps[i].append(float(gap))

    
    def get_valid_gaps(self):
        valid_gaps = {}
        for key, gaps in self.gaps.items():
            if min(gaps) > self.threshold:
                valid_gaps[key] = gaps
        return valid_gaps


    def get_comm_idle_time(self):
        valid_gaps = {}
        for key, gaps in self.gaps.items():
            min_gap = min(gaps)
            valid_gaps[key] = min_gap
        return valid_gaps
    

    def get_valid_comm_idle_time(self):
        valid_gaps = {}
        for key, gaps in self.gaps.items():
            min_gap = min(gaps)
            if min_gap > self.threshold:
                valid_gaps[key] = min_gap
        return valid_gaps


    def get_snapshot_strategy(self, block_sizes, bandwidth, local_rank_size, max_blocks=2):
        valid_gaps = self.get_valid_comm_idle_time()
        print(f"idel time: {self.get_comm_idle_time()}, valid gaps: {valid_gaps}")
        print("block sizes:", block_sizes)
        strategy = {}
        for key in valid_gaps.keys():
            strategy[key] = 0

        cur_block_id = 0
        total_blocks_num = len(block_sizes)

        for key, gap in valid_gaps.items():
            max_comm_size = gap / 1000 * bandwidth * (10**9) / 8 / 4 / local_rank_size * self.alpha
            blocks = 0
            cur_comm_size = 0
            while cur_block_id < total_blocks_num and cur_comm_size + block_sizes[cur_block_id] <= max_comm_size:
                cur_comm_size += block_sizes[cur_block_id]
                cur_block_id += 1
                blocks += 1
                if blocks >= max_blocks:
                    break

            strategy[key] = blocks
            if cur_block_id == total_blocks_num:
                break

        if cur_block_id < total_blocks_num:
            last_gap = len(self.get_comm_idle_time()) #max(valid_gaps.keys())
            # -1 indicates snapshot all the left blocks
            strategy[last_gap] = -1
            # TODO: if strategy[last_gap] = -1,
            # how to set the snapshot frequency to amortize the overhead?

        return strategy


    def plots(self):
        valid_gaps = self.get_valid_gaps()

        def plot(index, data):
            N = len(data)
            # sort the data in ascending order
            x = np.sort(data)
            # get the cdf values of y
            y = np.arange(N) / float(N)
            plt.plot(x, y, label=index)

            # min_value = min(data)
            # klargest = int(len(data) * 0.02)
            # max_value = nlargest(klargest, data)[-1]
            # plt.xlim(min_value, max_value)

        for index, gaps in valid_gaps.items():
            plot(index, gaps)

        plt.xlabel('Gap (ms)')
        plt.ylabel('CDF')
        plt.legend()
        plt.savefig(f'dist.png')


if __name__ == "__main__":
    import os
    filename = os.path.expanduser('~/zhuang/DeepSpeedExamples/bing_bert/snapshot_strategy/comm_gap.txt')
    snapshot_strategy = SnapshotStrategy(filename, threshold=10, alpha=0.8)
    # bandwidth = 100
    # block_size = 64 * 1024 * 1024
    # local_rank_size = 4
    # block_sizes = [block_size, block_size/1024, block_size, block_size/1024, block_size, block_size/1024, block_size, block_size]
    # strategy = snapshot_strategy.get_snapshot_strategy(block_sizes, bandwidth, local_rank_size)
    snapshot_strategy.plots()