
class GroupPolicy():
    def __init__(self, machines, gpus_per_machine, group_size=2):
        # only support group_size = 2 in this version
        assert(group_size == 2)
        self.machines = machines
        self.gpus_per_machine = gpus_per_machine
        self.group_size = group_size
        self.world_size = self.machines * self.gpus_per_machine
        self.set_machine_groups()
        self.set_rank_groups()
    

    def set_machine_groups(self):
        self.machine_groups = []
        assert(self.machines % self.group_size == 0)
        for i in range(0, self.machines, self.group_size):
            self.machine_groups.append(list(range(i, i+self.group_size)))


    def set_rank_groups(self):
        self.rank_groups = []
        for i in range(0, self.machines, self.group_size):
            self._set_rank_group(i)


    def _set_rank_group(self, machine_rank):
        for i in range(0, self.gpus_per_machine):
            group = []
            for j in range(machine_rank, machine_rank+self.group_size):
                group.append(j * self.gpus_per_machine + i)
            self.rank_groups.append(group)


    def get_machine_group(self, machine_rank):
        group_id = machine_rank // self.group_size
        return self.machine_groups[group_id]


    def get_machine_groups(self):
        return self.machine_groups


    def get_group_id(self, gpu_rank):
        machine_rank = gpu_rank // self.gpus_per_machine
        gpu_rank_on_machine = gpu_rank % self.gpus_per_machine
        return machine_rank // self.group_size * self.gpus_per_machine + gpu_rank_on_machine 


    def get_group(self, gpu_rank):
        return self.rank_groups[self.get_group_id(gpu_rank)]


    def get_groups(self):
        return self.rank_groups


    def get_group_size(self, rank):
        return self.group_size



if __name__ == "__main__":
    machines = 4
    gpus_per_machine = 8
    policy = GroupPolicy(machines, gpus_per_machine)
    for machine_id in range(machines):
        print(machine_id, policy.get_machine_group(machine_id))
    
    for rank_id in range(machines * gpus_per_machine):
        print(rank_id, policy.get_group(rank_id))