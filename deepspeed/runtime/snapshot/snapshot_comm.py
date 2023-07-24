"""batched collective operations for overhead amortization and better
bandwidth utilization"""

from .snapshot_group import SnapshotGroups
from .snapshot_record import SnapshotRecord
from .snapshot_global_variables import snapshot_profiler, snapshot_settings
import torch
import os
import json
import time
import typing
import pickle
import math
import signal
from typing import List
import atexit
from torch import Tensor
from deepspeed import comm as dist
from torch.cuda import Stream
from deepspeed.utils import instrument_w_nvtx, logger



def _snapshot_broadcast_obj(msg: typing.Any, rank: int, src = 0):
    """broadcast an arbitrary python object from ``src``.

    Note: ``msg`` must be pickleable.
    Args:
        msg (typing.Any): The object to broadcast.
        rank: GPU rank.
        src (int): Source rank.
    """
    if src == rank:
        # serialize the message
        msg_bc = pickle.dumps(msg)
        # construct a tensor to send
        msg_bc = torch.ByteTensor(torch.ByteStorage.from_buffer(msg_bc)).cuda()

        # broadcast meta and message
        length_tensor = torch.tensor([len(msg_bc)], dtype=torch.long).cuda()
        dist.broadcast(length_tensor, src=src)
        dist.broadcast(msg_bc, src=src)
        return msg
    else:
        # Get message meta
        length = torch.tensor([0], dtype=torch.long).cuda()
        dist.broadcast(length, src=src)

        # Receive and deserialize
        msg = torch.empty(length.item(), dtype=torch.uint8).cuda()
        dist.broadcast(msg, src=src)
        msg = pickle.loads(msg.cpu().numpy().tobytes())
        return msg



def _snapshot_torch_p2p_fn(input_tensor: Tensor,
                             output_tensor: Tensor,
                             snapshot_group: SnapshotGroups):
    this_rank = dist.get_rank()
    peer_rank = snapshot_group.get_peer_rank()

    # the performance of P2POp is very bad when G5 instances are used. what the fucc!
    ops = []
    send_op = torch.distributed.P2POp(torch.distributed.isend, input_tensor, peer_rank)
    ops.append(send_op)
    recv_op = torch.distributed.P2POp(torch.distributed.irecv, output_tensor, peer_rank)
    ops.append(recv_op)
    reqs = torch.distributed.batch_isend_irecv(ops)
    for req in reqs:
        req.wait()
    

def _snapshot_torch_allgather_fn(input_tensor: Tensor,
                             output_tensor: Tensor,
                             snapshot_group: SnapshotGroups):
    this_rank = dist.get_rank()
    comm_group = snapshot_group.get_snapshot_group()
    group_ranks = snapshot_group.get_snapshot_group_ranks()

    # for this time being, only consider one snapshot
    if this_rank == group_ranks[0]:
        output = torch.cat([torch.empty_like(input_tensor), output_tensor])
    else:
        output = torch.cat([output_tensor, torch.empty_like(input_tensor)])
    instrument_w_nvtx(dist.allgather_fn)(output, input_tensor, group=comm_group, async_op=True)
    

class SnapshotOptimizer():
    def __init__(self, group_size = 2, 
                dtype = torch.float, 
                snapshot_freq = 1,
                profile_rank = 0):
        # for this time being, only consider one snapshot
        assert group_size == 2
        self.snapshot_group_size = group_size
        self.profile_rank = profile_rank
        self.dtype = dtype
        self.snapshot_freq = snapshot_freq
        self.step_cnt = 0
        self.snapshot_versions = 2
        self.snapshot_current_version = 0
        self.snapshot_gpu_versions = 4
        self.snapshot_gpu_buffer_id = 0
        self.cur_block_id = 0
        self.total_blocks = 0
        self.comm_gap_id = 0
        self.snapshot_mode = "naive"
        self.have_set_snapshot_strategy = False
        self.snapshot_strategy = dict()
        self.local_optimizer_state_dict = None
        self.checkpoint_comm_stream = Stream()
        self.checkpoint_copy_stream = Stream()
        self.allgather_stream = None
        self.reducescatter_stream = None


    def set_checkpoint_setup(self, args):
        # NVLink, one instance, 175Gbps per GPU
        self.bandwidth = args.network_bandwidth
        # for NVLink, choose 32 * 1024 * 1024 (FP32) as the buffer size
        # for 100Gbps network, choose 2 (1) * 1024 * 1024  (FP32) (or even smaller) as the buffer size
        # for 400Gbps network, choose xxx
        self.snapshot_buffer_size = args.snapshot_buffer_size * 1024 * 1024
        self.span_threshold = args.span_threshold
        self.span_alpha = args.span_alpha
        self.max_blocks_in_span = args.max_blocks_in_span
        self.save_to_disk = args.save_to_disk
        self.dir = args.snapshot_path
        self.jump_profile_lines = args.jump_profile_lines
        # for test only
        # self.dir = '/home/ec2-user/zhuang/DeepSpeedExamples/bing_bert/saved_models/bing_bert_base_seq/snapshot'
        # self.dir = "/tmp/ramdisk/snapshot" # RamDisk
        os.makedirs(self.dir, exist_ok=True)



    def init_snapshot_comm_groups(self):
        self.this_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_size = dist.get_local_size()
        # for single-machine test only
        if self.local_size == self.world_size:
            self.local_size = 1
        assert(self.world_size % self.local_size == 0)
        self.machines = self.world_size // self.local_size
        self.snapshot_group = SnapshotGroups(self.machines, self.local_size, self.snapshot_group_size)
        self.peer_rank = self.snapshot_group.get_peer_rank()


    def is_snapshot_step(self):
        return self.step_cnt % self.snapshot_freq == 0 and self.have_set_snapshot_strategy


    @torch.no_grad()
    def init_optimizer_state_on_cpu(self, optimizer_state_dict):
        self.snapshot_record = SnapshotRecord(self.dir, self.this_rank)
        tmp_file = f"tmp_checkpoint_{self.this_rank}.pt"
        torch.save(optimizer_state_dict, tmp_file)

        self.snapshot_gpu_buffers = [torch.zeros(self.snapshot_buffer_size, dtype = self.dtype, device=torch.cuda.current_device())] * self.snapshot_gpu_versions
        # we use multiple versions in memory to store the checkpoints
        self.optimizer_state_dicts = [[] for _ in range(self.snapshot_versions)]
        self.optimizer_tensors = [[] for _ in range(self.snapshot_versions)]
        self.tensor_blocks = [[] for _ in range(self.snapshot_versions)]
        self.tensor_sizes = []
        self.block_sizes = [[] for _ in range(self.snapshot_versions)] 
        self.block_nums = [0 for _ in range(self.snapshot_versions)]

        for v in range(self.snapshot_versions):
            self.optimizer_state_dicts[v] = torch.load(tmp_file, map_location=torch.device('cpu'))
            optimizer_state = self.optimizer_state_dicts[v]['optimizer_state_dict']
            states = optimizer_state['state']
            param_groups = optimizer_state['param_groups']
            fp32_flat_groups = self.optimizer_state_dicts[v]['fp32_flat_groups']

            for group in param_groups:
                for p in group['params']:
                    state = states[p]
                    self.optimizer_tensors[v].append(state['exp_avg'].pin_memory())
                    self.optimizer_tensors[v].append(state['exp_avg_sq'].pin_memory())
                    self.optimizer_tensors[v].append(fp32_flat_groups[p].pin_memory())
        
            for tensor in self.optimizer_tensors[v]:
                numel = tensor.numel()
                if v == 0:
                    self.tensor_sizes.append(numel)
                blocks_num = math.ceil(numel / self.tensor_block_size)
                self.block_nums[v] += blocks_num
                for i in range(0, blocks_num):
                    start_pos = i * self.tensor_block_size
                    end_pos = min((i+1) * self.tensor_block_size, numel)
                    self.tensor_blocks[v].append(tensor[start_pos: end_pos])
                    self.block_sizes[v].append(end_pos - start_pos)

        snapshot_profiler.set_checkpoint_data_size(sum(self.tensor_sizes))
        logger.info(f"[Rank {self.this_rank}] optimizer size: {snapshot_profiler.get_checkpoint_data_size_in_MB()} MB")
        os.system(f"rm {tmp_file}")



    def set_snapshot_mode(self, mode):
        """
        there are two modes of snapshot: naive and interleave
        naive: block computation to snapshot all blocks before the next iteration
        interleave: interleave snapshot with training traffic to minimize the overhead
        """
        if mode == "interleave":
            self.snapshot_mode = mode
            self.have_set_snapshot_strategy = False
        

    def extract_states_for_snapshot(self):
        if not self.is_snapshot_step() or self.local_optimizer_state_dict is None:
            return

        optimizer_state = self.local_optimizer_state_dict['optimizer_state_dict']
        states = optimizer_state['state']
        param_groups = optimizer_state['param_groups']
        fp32_flat_groups = self.local_optimizer_state_dict['fp32_flat_groups']
        snapshot_tensors = []
        for group in param_groups:
            for p in group['params']:
                state = states[p]
                snapshot_tensors.append(state['exp_avg'])
                snapshot_tensors.append(state['exp_avg_sq'])
                snapshot_tensors.append(fp32_flat_groups[p])

        self.snapshot_blocks = []
        for tensor in snapshot_tensors:
            numel = tensor.numel()
            blocks_num = math.ceil(numel / self.tensor_block_size)

            for i in range(0, blocks_num):
                start_pos = i * self.tensor_block_size
                end_pos = min((i+1) * self.tensor_block_size, numel)
                self.snapshot_blocks.append(tensor[start_pos: end_pos])
        self.total_blocks = len(self.snapshot_blocks)


           
    def reset_cpu_buffer(self):
        self.cur_block_id = 0
        self.snapshot_current_version = (self.snapshot_current_version + 1) % self.snapshot_versions


    def move_to_cpu(self, tensor: Tensor, tensor_on_cpu):
        tensor_on_cpu.copy_(tensor, non_blocking=True)


    def get_snapshot_tensor(self, tensor_id):
        return self.optimizer_tensors[self.snapshot_current_version][tensor_id]


    def get_snapshot_tensor_size(self, tensor_id):
        return self.tensor_sizes[self.snapshot_current_version][tensor_id]


    def get_snapshot_block(self, block_id):
        return self.tensor_blocks[self.snapshot_current_version][block_id]


    def get_snapshot_block_size(self, block_id):
        return self.block_sizes[self.snapshot_current_version][block_id]


    def snapshot(self, input_tensor: Tensor):
        block_id = self.cur_block_id
        block_size = self.get_snapshot_block_size(block_id)
        snapshot_gpu_buffer = self.snapshot_gpu_buffers[self.snapshot_gpu_buffer_id]
        output_tensor = snapshot_gpu_buffer[:block_size]
        self.snapshot_gpu_buffer_id = (self.snapshot_gpu_buffer_id + 1) % self.snapshot_gpu_versions

        self.cur_block_id += 1

        _snapshot_torch_p2p_fn(input_tensor, output_tensor, self.snapshot_group)
        # _snapshot_torch_p2p_fn(input_tensor, output_tensor, self.snapshot_group)
        # with torch.cuda.stream(self.checkpoint_copy_stream):
        #     # snapshot_profiler.record_gpu_cpu_copy_start_time(self.checkpoint_copy_stream)
        #     tensor_on_cpu = self.get_snapshot_block(block_id)
        #     self.move_to_cpu(output_tensor, tensor_on_cpu)
        #     #snapshot_profiler.record_gpu_cpu_copy_end_time(self.checkpoint_copy_stream)


    def set_allgather_stream(self, stream):
        if not self.allgather_stream:
            self.allgather_stream = stream


    def set_reducescatter_stream(self, stream):
        if not self.reducescatter_stream:
            self.reducescatter_stream = stream


    def remote_snapshot(self, event=None, stream=None):
        if self.is_snapshot_step():
            if self.snapshot_mode == "interleave" and self.comm_gap_id in self.snapshot_strategy:
                blocks = self.snapshot_strategy[self.comm_gap_id]
                self.remote_snapshot_blocks(blocks, event, stream)
            self.comm_gap_id += 1


    @instrument_w_nvtx
    def remote_snapshot_blocks(self, blocks, event, stream):
        if blocks == -1:
            # snapshot the all the left blocks
            blocks = self.total_blocks - self.cur_block_id
            print(f"{blocks} blocks in the last span")
            return

        with torch.cuda.stream(self.checkpoint_comm_stream):
            if event is not None:
                event.wait()
            if snapshot_settings.is_profile_mode():
                snapshot_profiler.record_checkpoint_comm_start_time(self.checkpoint_comm_stream)
                for _ in range(blocks):
                    self.snapshot_block()
                snapshot_profiler.record_checkpoint_comm_end_time(self.checkpoint_comm_stream)
            else:
                for _ in range(blocks):
                    self.snapshot_block()
            end_event = torch.cuda.Event(enable_timing=False)
            end_event.record(stream=self.checkpoint_comm_stream)
            end_event.wait(stream=stream)
        self.reducescatter_stream.wait_stream(self.checkpoint_comm_stream)
        # stream.wait_stream(self.checkpoint_comm_stream)
        


    def snapshot_block(self):
        tensor = self.snapshot_blocks[self.cur_block_id]
        self.snapshot(tensor)


    @instrument_w_nvtx
    def snapshot_all_blocks(self):
        if self.is_snapshot_step():
            with torch.cuda.stream(self.checkpoint_comm_stream):
                for tensor in self.snapshot_blocks:
                    self.snapshot(tensor)


    def set_remote_snapshot_name(self):
        return os.path.join(self.dir, f"{self.this_rank}_{self.peer_rank}.pt")


    def set_local_snapshot_name(self):
        return os.path.join(self.dir, f"{self.this_rank}.pt")


    def get_optimizer_step(self, optimizer_state):
        param_groups = optimizer_state['optimizer_state_dict']['state']
        return param_groups[0]['step']


    def update_optimizer_state_dict(self, optimizer_state):
        if self.is_snapshot_step():
            step = self.get_optimizer_step(optimizer_state)
            # update the step information in self.optimizer_state_dict
            remote_optimizer_state = self.optimizer_state_dicts[self.snapshot_current_version]['optimizer_state_dict']
            states = remote_optimizer_state['state']
            param_groups = remote_optimizer_state['param_groups']

            for group in param_groups:
                for p in group['params']:
                    states[p]['step'] = step


    def update_local_optimizer_snapshot(self, optimizer_state): 
        self.local_optimizer_state_dict = optimizer_state
        self.step_cnt += 1
        self.comm_gap_id = 0
        self.reset_cpu_buffer()

        if self.snapshot_mode == "naive" and not self.have_set_snapshot_strategy:
            self.have_set_snapshot_strategy = True
            self.tensor_block_size = self.snapshot_buffer_size
            self.init_snapshot_comm_groups()
            self.init_optimizer_state_on_cpu(optimizer_state)
            self.extract_states_for_snapshot()
        elif self.snapshot_mode == "interleave" and \
            not self.have_set_snapshot_strategy and \
            snapshot_settings.is_comm_profile_finished():
            self.have_set_snapshot_strategy = True
            self.init_snapshot_comm_groups()
            self.tensor_block_size = self.snapshot_buffer_size
            self.init_optimizer_state_on_cpu(optimizer_state)
            snapshot_strategy = self.compute_snapshot_strategy()
            self.snapshot_strategy = _snapshot_broadcast_obj(snapshot_strategy, self.this_rank, src=self.profile_rank)
            torch.cuda.synchronize()
            logger.info(f"[Rank {self.this_rank}] snapshot strategy: {self.snapshot_strategy}")
            self.extract_states_for_snapshot()

        # self.update_optimizer_state_dict(optimizer_state)
        if self.snapshot_mode == "naive" and self.is_snapshot_step():
            self.snapshot_all_blocks()
        

    def compute_snapshot_strategy(self):
        if self.this_rank == self.profile_rank:
            from .snapshot_global_variables import comm_profiler
            from .snapshot_strategy import SnapshotStrategy
            filename = comm_profiler.get_profile_filename()
            print("compute_snapshot_strategy", filename)
            snapshot_strategy = SnapshotStrategy(filename, threshold=self.span_threshold, jump_lines=self.jump_profile_lines, alpha=self.span_alpha)
            return snapshot_strategy.get_snapshot_strategy(self.block_sizes[0], self.bandwidth, self.local_size, self.max_blocks_in_span)
        else:
            return None


    def latest_completed_snapshot_version(self):
        return (self.snapshot_current_version - 1) % self.snapshot_versions


    def save_optimizer_snapshot_to_disk(self):
        if self.local_optimizer_state_dict is None:
            return 
        logger.info(f"[Rank {self.this_rank}] save optimizer snapshot to disk ...")
        
        if self.save_to_disk:
            # local snapshot
            torch.cuda.synchronize()
            start_time = time.time()
            optimizer_size = snapshot_profiler.get_checkpoint_data_size_in_MB()
            local_zero_sd = dict(optimizer_state_dict=self.local_optimizer_state_dict)
            local_snapshot_filename = self.set_local_snapshot_name()
            torch.save(local_zero_sd, local_snapshot_filename)
            torch.cuda.synchronize()
            logger.info(f"""[Rank {self.this_rank}] save optimizer on GPU to disk \n
                    snapshot filename: {local_snapshot_filename} \n
                    Checkpoint size: {optimizer_size} MB \n
                    time: {time.time() - start_time}""")
            # remote snapshot
            torch.cuda.synchronize()
            start_time = time.time()
            last_complete_version = self.latest_completed_snapshot_version()
            remote_zero_sd = dict(optimizer_state_dict=self.optimizer_state_dicts[last_complete_version])
            remote_snapshot_filename = self.set_remote_snapshot_name()
            torch.save(remote_zero_sd, remote_snapshot_filename)
            torch.cuda.synchronize()
            logger.info(f"""[Rank {self.this_rank}] save optimizer on CPU to disk \n
                    snapshot filename: {local_snapshot_filename} \n
                    Checkpoint size: {optimizer_size} MB \n
                    time: {time.time() - start_time}""")
            record = self.snapshot_record.construct_record(remote_snapshot_filename)
            self.snapshot_record.write(record)



    # manually save the snapshot to disk periodically
    def save_optimizer_snapshot(self):
        if self.is_snapshot_step():
            self.reset_cpu_buffer()
            # local snapshot
            local_zero_sd = dict(optimizer_state_dict=self.local_optimizer_state_dict)
            local_snapshot_name = self.set_local_snapshot_name()
            torch.save(local_zero_sd, local_snapshot_name)
            # remote snapshot
            last_complete_version = self.latest_completed_snapshot_version()
            remote_zero_sd = dict(optimizer_state_dict=self.optimizer_state_dicts[last_complete_version])
            snapshot_filename = self.set_remote_snapshot_name()
            torch.save(remote_zero_sd, snapshot_filename)
            record = self.snapshot_record.construct_record(snapshot_filename)
            self.snapshot_record.write(record)


cpu_snapshot = SnapshotOptimizer()
signal.signal(signal.SIGINT, cpu_snapshot.save_optimizer_snapshot_to_disk)
signal.signal(signal.SIGTERM, cpu_snapshot.save_optimizer_snapshot_to_disk)
atexit.register(cpu_snapshot.save_optimizer_snapshot_to_disk)