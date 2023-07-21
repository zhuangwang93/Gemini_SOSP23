"""unit tests for coalesced collectives"""

import torch
import deepspeed.comm as dist
from deepspeed.runtime.snapshot.snapshot_comm import SnapshotComm
from time import sleep

from .common import distributed_test



def init_tensors(tensor_shapes, dtype=torch.float):
    rank = dist.get_local_rank()
    input_tensors = []
    for shape in tensor_shapes:
        tensor = torch.ones(shape, dtype = dtype, device=torch.cuda.current_device()) * (rank + 1)
        input_tensors.append(tensor)
    return input_tensors


@distributed_test(world_size=4)
def test_group_snapshot_same_size():
    rank = dist.get_local_rank()
    tensor_num = 4
    tensor_shape = [4]
    tensor_shapes = [tensor_shape] * tensor_num
    print(tensor_shapes, flush=True)
    snapshot_comm = SnapshotComm(tensor_shapes, tensor_shapes)
    input_tensors = init_tensors(tensor_shapes)
    print(f"{rank} init", input_tensors, flush=True)
    for tensor in input_tensors:
        snapshot_comm.group_snapshot(tensor)
    tensors_on_cpu = snapshot_comm.decode_cpu_buffer()
    print(f"{rank} after snapshot", tensors_on_cpu, flush=True)


@distributed_test(world_size=4)
def test_group_snapshot_save():
    rank = dist.get_local_rank()
    tensor_num = 4
    tensor_shape = [4]
    tensor_shapes = [tensor_shape] * tensor_num
    print(tensor_shapes, flush=True)
    snapshot_comm = SnapshotComm(tensor_shapes, tensor_shapes)
    input_tensors = init_tensors(tensor_shapes)
    print(f"{rank} init", input_tensors, flush=True)

    def comm(snapshot_comm, input_tensors):
        for tensor in input_tensors:
            snapshot_comm.group_snapshot(tensor)

    comm(snapshot_comm, input_tensors)
    cpt_filename = f"snapshot/{rank}_test.pt"
    snapshot_comm.save_snapshot(filename=cpt_filename)


@distributed_test(world_size=4)
def test_group_snapshot_load():
    rank = dist.get_local_rank()
    tensor_num = 4
    tensor_shape = [4]
    tensor_shapes = [tensor_shape] * tensor_num
    snapshot_comm = SnapshotComm(tensor_shapes, tensor_shapes)
    cpt_filename = f"snapshot/{rank}_test.pt"
    tensors = snapshot_comm.load_snapshot(filename=cpt_filename, mode="local")
    print(f"{rank} local load", tensors, flush=True)

    if rank % 2 == 0:
        mode = "recv"
        peer = rank + 1
    else:
        mode = "send"
        peer = rank - 1
    tensors = snapshot_comm.load_snapshot(filename=cpt_filename, mode=mode, peer=peer)
    print(f"{rank} remote {mode} {peer}", tensors, flush=True)


@distributed_test(world_size=4)
def test_group_snapshot_coalesced_same_size():
    rank = dist.get_local_rank()
    tensor_num = 4
    tensor_shape = [4]
    tensor_shapes = [tensor_shape] * tensor_num
    print(tensor_shapes, flush=True)
    snapshot_comm = SnapshotComm(tensor_shapes, tensor_shapes)
    input_tensors = init_tensors(tensor_shapes)
    print(f"{rank} init", input_tensors, flush=True)
    tensors_coalesced = 2
    for i in range(0, tensor_num, tensors_coalesced):
        tensors = input_tensors[i:i+tensors_coalesced]
        snapshot_comm.group_snapshot_coalesced(tensors)
    tensors_on_cpu = snapshot_comm.decode_cpu_buffer()
    print(f"{rank} after snapshot", tensors_on_cpu, flush=True)


@distributed_test(world_size=4)
def test_group_snapshot_different_size():
    rank = dist.get_local_rank()
    tensor_num = 4
    if rank % 2 == 0:
        send_tensor_shape = [4]
        recv_tensor_shape = [6]
    else:
        send_tensor_shape = [6]
        recv_tensor_shape = [4]
    send_tensor_shapes = [send_tensor_shape] * tensor_num
    recv_tensor_shapes = [recv_tensor_shape] * tensor_num
    print(send_tensor_shapes, recv_tensor_shapes, flush=True)

    snapshot_comm = SnapshotComm(send_tensor_shapes, recv_tensor_shapes)
    input_tensors = init_tensors(send_tensor_shapes)

    print(f"{rank} init", input_tensors, flush=True)
    for tensor in input_tensors:
        snapshot_comm.group_snapshot(tensor)
    tensors_on_cpu = snapshot_comm.decode_cpu_buffer()
    print(f"{rank} after snapshot", tensors_on_cpu, flush=True)


@distributed_test(world_size=4)
def test_group_snapshot_coalesced_different_size():
    rank = dist.get_local_rank()
    tensor_num = 4
    if rank % 2 == 0:
        send_tensor_shape = [2, 2]
        recv_tensor_shape = [3, 2]
    else:
        send_tensor_shape = [3, 2]
        recv_tensor_shape = [2, 2]
    send_tensor_shapes = [send_tensor_shape] * tensor_num
    recv_tensor_shapes = [recv_tensor_shape] * tensor_num
    print(send_tensor_shapes, recv_tensor_shapes, flush=True)

    snapshot_comm = SnapshotComm(send_tensor_shapes, recv_tensor_shapes)
    input_tensors = init_tensors(send_tensor_shapes)

    print(f"{rank} init", input_tensors, flush=True)
    tensors_coalesced = 2
    for i in range(0, tensor_num, tensors_coalesced):
        tensors = input_tensors[i:i+tensors_coalesced]
        snapshot_comm.group_snapshot_coalesced(tensors)
    tensors_on_cpu = snapshot_comm.decode_cpu_buffer()
    print(f"{rank} after snapshot", tensors_on_cpu, flush=True)