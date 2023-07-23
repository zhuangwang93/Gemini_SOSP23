import time
import torch
import os
from deepspeed.utils import logger


class TrainingProfiler():
    def __init__(self):
        self.training_start_time = time.time()
        self.training_init_time = 0

        self.checkpoint_load_start_time = 0
        self.checkpoint_load_end_time = 0
        self.checkpoint_load_time = 0

        self.checkpoint_recover_start_time = 0
        self.checkpoint_recover_end_time = 0
        self.checkpoint_recover_time = 0
        

    def set_training_init_time(self):
        self.training_init_time = time.time() - self.training_start_time


    def get_training_init_time(self):
        return self.training_init_time


    def record_checkpoint_load_start_time(self):
        self.checkpoint_load_start_time = time.time()

    
    def record_checkpoint_load_end_time(self):
        self.checkpoint_load_end_time = time.time()
        self.checkpoint_load_time = self.checkpoint_load_end_time - self.checkpoint_load_start_time


    def get_checkpoint_load_time(self):
        return self.checkpoint_load_time


    def record_checkpoint_recover_start_time(self):
        self.checkpoint_recover_start_time = time.time()


    def record_checkpoint_recover_end_time(self):
        self.checkpoint_recover_end_time = time.time()
        self.checkpoint_recover_time = self.checkpoint_recover_end_time - self.checkpoint_recover_start_time


    def get_checkpoint_recover_time(self):
        return self.checkpoint_recover_time


    def report(self):
        logger.info(f"""
                training init time: {self.training_init_time}
                checkpoint load time: {self.checkpoint_load_time}
                checkpoint recovery time: {self.checkpoint_recover_time}
                """)



class SnapshotProfiler():
    def __init__(self):
        self.last_iteration_end_time = 0
        self.iteration_time = 0
        self.checkpoint_data_size = 0

        self.checkpoint_comm_start_time = 0
        self.checkpoint_comm_time = []

        self.gpu_cpu_copy_start_time = 0
        self.gpu_cpu_copy_time = []

    
    def reset(self):
        self.checkpoint_comm_time = []
        self.gpu_cpu_copy_time = []


    def synchronize(self):
        torch.cuda.synchronize()


    def record_iteration_end_time(self):
        cur_time = time.time()
        if self.last_iteration_end_time > 0:
            self.iteration_time = cur_time - self.last_iteration_end_time
        self.last_iteration_end_time = cur_time

    
    def set_iteration_time(self, iteration_time):
        self.iteration_time = iteration_time


    def get_iteration_time(self):
        return self.iteration_time


    def set_checkpoint_data_size(self, size):
        self.checkpoint_data_size = size

    
    def get_checkpoint_data_size(self):
        return self.checkpoint_data_size


    def get_checkpoint_data_size_in_MB(self):
        return self.checkpoint_data_size * 4 / 1024 // 1024


    def record_checkpoint_comm_start_time(self, stream):
        stream.synchronize()
        self.checkpoint_comm_start_time = time.time()


    def record_checkpoint_comm_end_time(self, stream):
        stream.synchronize()
        checkpoint_comm_time = time.time() - self.checkpoint_comm_start_time
        self.checkpoint_comm_time.append(checkpoint_comm_time)


    def get_checkpoint_comm_time(self):
        return self.checkpoint_comm_time


    def record_gpu_cpu_copy_start_time(self, stream):
        stream.synchronize()
        self.gpu_cpu_copy_start_time = time.time()


    def record_gpu_cpu_copy_end_time(self, stream):
        stream.synchronize()
        gpu_cpu_copy_time = time.time() - self.gpu_cpu_copy_start_time
        self.gpu_cpu_copy_time.append(gpu_cpu_copy_time)


    def get_gpu_cpu_copy_time(self):
        return self.checkpoint_comm_time


    def report(self):
        if sum(self.checkpoint_comm_time) > 0:
            logger.info(f"""
                    iteration time: {self.iteration_time}
                    checkpoint data size: {self.get_checkpoint_data_size_in_MB()} MB
                    checkpoint communication time: {sum(self.checkpoint_comm_time)}
                    GPU-to-CPU copy time: {sum(self.gpu_cpu_copy_time)}
                    """)



class CommounicationProfiler():
    def __init__(self, dir="snapshot_strategy", filename="comm_gap.txt"):
        self.start_allgather_events = []
        self.end_allgather_events = []

        self.start_reducescatter_events = []
        self.end_reducescatter_events = []

        os.makedirs(dir, exist_ok=True)
        self.filename = os.path.join(dir, filename)
        # with open(self.filename, 'w') as f:
        #     f.write("")


    def add_start_allgather_event(self, event):
        self.start_allgather_events.append(event)


    def add_end_allgather_event(self, event):
        self.end_allgather_events.append(event)


    def add_start_reducescatter_event(self, event):
        self.start_reducescatter_events.append(event)


    def add_end_reducescatter_event(self, event):
        self.end_reducescatter_events.append(event)


    def set_profile_filename(self, filename):
        self.filename = filename

    
    def get_profile_filename(self):
        return self.filename
        

    def reset_comm_event(self):
        self.start_allgather_events = []
        self.end_allgather_events = []

        self.start_reducescatter_events = []
        self.end_reducescatter_events = []

        self.comm_gap_id = 0


    def profile_comm_time_gap(self):
        torch.cuda.synchronize()
        last_end_event = None
        comm_times = []
        total_comm_time = 0
        with open(self.filename, 'a+') as f:
            # allgather event
            # for start_event, end_event in zip(self.start_allgather_events, self.end_allgather_events):
            #     comm_time = round(start_event.elapsed_time(end_event), 2)
            #     comm_times.append(comm_time)
            #     total_comm_time += comm_time

            #     if last_end_event is None:
            #         last_end_event = end_event
            #         continue
            #     else:
            #         time_gap = last_end_event.elapsed_time(start_event)
            #         f.write(str(round(time_gap, 1)) + " ")
            #         last_end_event = end_event

            # allgather event
            # reverse the communication time and the idle time span. Cannot understand
            for start_event, end_event in zip(self.start_allgather_events, self.end_allgather_events):
                time_gap = round(start_event.elapsed_time(end_event), 2)
                f.write(str(round(time_gap, 1)) + " ")

                if last_end_event is None:
                    last_end_event = end_event
                    continue
                else:
                    comm_time = round(last_end_event.elapsed_time(start_event), 2)
                    comm_times.append(comm_time)
                    total_comm_time += comm_time
                    last_end_event = end_event

            # # f.write(" ")
            # # reducescatter event
            # comm_times.append("reducescatter:")

            for start_event, end_event in zip(self.start_reducescatter_events, self.end_reducescatter_events):
                comm_time = round(start_event.elapsed_time(end_event), 2)
                comm_times.append(comm_time)
                total_comm_time += comm_time

                if last_end_event is None:
                    last_end_event = end_event
                    continue

                time_gap = last_end_event.elapsed_time(start_event)
                if time_gap < 0:
                    time_gap = 0
                f.write(str(round(time_gap, 1)) + " ")
                last_end_event = end_event
            f.write("\n")
        print(f"[total comm time] {total_comm_time} {comm_times}")
        self.reset_comm_event()