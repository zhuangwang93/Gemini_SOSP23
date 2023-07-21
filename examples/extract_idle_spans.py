
class NetworkIdleSpans():
    def __init__(self, filename, threshold=200, jump_lines=6):
        self.filename = filename
        self.threshold = threshold
        self.jump_lines = jump_lines
        self.extract_data()


    def extract_data(self):
        self.idle_spans = []
        with open(self.filename, "r") as f:
            for line in f:
                # jump the first few lines 
                if self.jump_lines > 0:
                    self.jump_lines -= 1
                    continue

                idle_spans = []
                gaps = line.strip().split(" ")
                print(gaps)
                for gap in gaps:
                    idle_spans.append(float(gap))
                self.idle_spans.append(idle_spans)



    def avg(self, list):
        if len(list) == 0:
            return 0
        return round(sum(list) / len(list), 2)


    def get_idle_spans(self):
        if len(self.idle_spans) == 0:
            return []
        
        idle_spans_num = len(self.idle_spans[0])
        self.idle_spans_column = [[] for _ in range(idle_spans_num)]
        self.avg_idle_spans = []
        for idle_spans in self.idle_spans:
            for i, span in enumerate(idle_spans):
                self.idle_spans_column[i].append(span)

        for idle_spans_column in self.idle_spans_column:
            self.avg_idle_spans.append(min(idle_spans_column))
        return self.avg_idle_spans

                

    def get_total_idle_time(self):
        total_idle_time = 0
        for span in self.avg_idle_spans:
            if span >= self.threshold:
                total_idle_time += span
        return total_idle_time



if __name__ == "__main__":
    import os
    filename = os.path.expanduser('GPT/snapshot_strategy/comm_gap.txt')
    network_idle_spans = NetworkIdleSpans(filename, threshold=200)
    avg_idle_spans = network_idle_spans.get_idle_spans()
    total_idle_time = network_idle_spans.get_total_idle_time()
    print(avg_idle_spans)
    print(total_idle_time)