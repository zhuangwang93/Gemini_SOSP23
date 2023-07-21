import torch
from deepspeed import comm as dist
from .snapshot_policy import GroupPolicy


class SnapshotGroups():
    def __init__(self, machines, gpus_per_machine, group_size=2):
        self.rank = dist.get_rank()
        self.group_policy = GroupPolicy(machines, gpus_per_machine, group_size)
        self.set_snapshot_groups()


    def set_snapshot_groups(self):
        self.rank_groups = self.group_policy.get_groups()
        self.snapshot_groups = [torch.distributed.new_group(group) for group in self.rank_groups]


    def get_snapshot_group(self):
        group_id = self.group_policy.get_group_id(self.rank)
        return self.snapshot_groups[group_id]


    def get_snapshot_group_ranks(self):
        return self.group_policy.get_group(self.rank)


    def get_snapshot_group_size(self):
        return self.group_policy.get_group_size(self.rank)


    def get_peer_rank(self):
        # for this time being, only consider a group size of two
        group_ranks = self.get_snapshot_group_ranks()
        for rank in group_ranks:
            if rank != self.rank:
                self.peer_rank = rank
                return self.peer_rank