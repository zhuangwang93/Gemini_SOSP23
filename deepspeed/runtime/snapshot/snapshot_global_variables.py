from .snapshot_profiler import TrainingProfiler, SnapshotProfiler, CommounicationProfiler


class SnapshotSettings():
    def __init__(self):
        self.snapshot_mode = "none"
        self.is_cpu_snapshot = False
        self.profile_mode = False
        self.comm_profile_mode = False
        self.global_step = 0
        self.global_epoch = 0
        self.comm_profile_steps = 0
        self.comm_profile_finished = False


    def set_global_step(self, step = 0):
        self.global_step = step
        if self.get_global_step() >= self.comm_profile_steps:
            if self.comm_profile_mode:
                self.comm_profile_finished = True
            self.comm_profile_mode = False


    def set_global_epoch(self, epoch = 0):
        self.global_epoch = epoch


    def is_comm_profile_finished(self):
        return self.comm_profile_finished 


    def increment_step(self):
        self.global_step += 1


    def get_global_step(self):
        return self.global_step


    def get_global_epoch(self):
        return self.global_epoch


    def set_comm_profile_steps(self, steps = 20):
        self.comm_profile_steps = steps
        if steps > 0:
            self.comm_profile_mode = True


    def set_profile_mode(self, mode):
        self.profile_mode = mode


    def is_profile_mode(self):
        return self.is_snapshot_mode() and self.profile_mode


    def set_snapshot_mode(self, mode):
        """
        There are three snapshot mode: none, naive, and interleave
        none: no snapshot
        naive: block computation to snapshot all blocks before the next iteration
        interleave: interleave snapshot with training traffic to minimize the overhead
        """
        self.snapshot_mode = mode
        self.is_cpu_snapshot = self.snapshot_mode in ("naive", "interleave")


    def get_snapshot_mode(self):
        return self.snapshot_mode


    def is_snapshot_mode(self):
        return self.is_cpu_snapshot


    def set_comm_profile_mode(self, mode):
        self.comm_profile_mode = mode


    def is_comm_profile_mode(self):
        return self.comm_profile_mode


snapshot_settings = SnapshotSettings()
training_profiler = TrainingProfiler()
snapshot_profiler = SnapshotProfiler()
comm_profiler = CommounicationProfiler()