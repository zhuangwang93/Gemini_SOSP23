from mpi4py import MPI
import torch
import argparse
import os
import json


class RecoverSnapshot():
    def __init__(self, args):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.world_size = self.comm.Get_size()
        self.gpus_per_machine = self.world_size // 2
        self.machine_id = self.rank // self.gpus_per_machine
        self.local_rank = self.rank % self.gpus_per_machine
        
        self.path = args.path
        self.dir = args.dir
        # the rank of the source snapshot during training. Its machine_id is 0 
        self.src = args.src
        # the rank of the destination snapshot during training. Its machine_id is 1
        self.dst = args.dst


    def extract_snapshot_info(self, training_rank):
        filename = os.path.join(self.path, self.dir, f"{training_rank}.json")
        with open(filename, "r") as f:
            json_data = json.load(f)
        return json_data["iteration"], json_data["epoch"], json_data["file"]


    def get_checkpoint_name_template(self):
        filename = os.path.join(self.path, "checkpoint_filename")
        with open(filename, "r") as f:
            template = json.load(f)
        return template


    def set_optim_checkpoint_name_from_template(self, template, rank):
        words = template.split("_")
        words[3] = str(rank)
        return '_'.join(words)

    
    def get_model_checkpoint_name(self, optim_filename):
        return optim_filename.replace("optim", "model")


    def recover_GPU_snapshot(self):
        """
        There is only one sender machine and one receiver machine for the snapshot, 
        but we use all GPUs on each machine for communications.
        """
        src, dst = self.local_rank, self.local_rank + self.gpus_per_machine
        if self.rank == src:
            training_rank = self.local_rank + self.src * self.gpus_per_machine
            # communicate the filename template of checkpoint for recovery
            ckpt_template = self.get_checkpoint_name_template()
            self.comm.send(ckpt_template, dest=dst)
            iteration, epoch, filename = self.extract_snapshot_info(training_rank)

            # communicate the checkpoint of optimizer states
            data = torch.load(filename)
            self.comm.send(data, dest=dst)

            # rename the filename of local checkpoint for recovery
            ckpt_filename = self.set_optim_checkpoint_name_from_template(ckpt_template, training_rank)
            local_filename = f"{training_rank}.pt"
            local_abs_filename = os.path.join(self.path, self.dir, local_filename)
            ckpt_abs_filename = os.path.join(self.path, self.dir, ckpt_filename)
            os.rename(local_abs_filename, ckpt_abs_filename)

            # communicate the checkpoint of model states
            model_states_filename = self.get_model_checkpoint_name(ckpt_abs_filename)
            data = torch.load(model_states_filename)
            # update the latest iteration number
            # TODO: the iteration info is different from the step information recorded from deepspeed_train.py
            data["global_steps"] = iteration
            data["last_global_step"] = iteration
            data["epoch"] = epoch
            self.comm.send(data, dest=dst)
            torch.save(data, model_states_filename)
        elif self.rank == dst:
            ckpt_template = self.comm.recv(source=src)
            training_rank = self.local_rank + self.dst * self.gpus_per_machine
            # set the filename of checkpoint for recovery
            filename = self.set_optim_checkpoint_name_from_template(ckpt_template, training_rank)
            path = os.path.join(self.path, self.dir)
            os.makedirs(path, exist_ok=True)
            filename = os.path.join(path, filename)
            # recv checkpoints from other GPU machines
            data = self.comm.recv(source=src)
            torch.save(data, filename)

            # communicate the checkpoint of model states
            model_states_filename = self.get_model_checkpoint_name(filename)
            data = self.comm.recv(source=src)
            torch.save(data, model_states_filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='snapshot recovery',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', '-p', type=str, default="/home/ec2-user/zhuang/DeepSpeedExamples/bing_bert/saved_models/bing_bert_base_seq",
                        help='the path for snapshots')
    parser.add_argument('--dir', type=str, default="snapshot",
                        help='the directory for snapshots')
    parser.add_argument('--src', '-s', type=int, default=0,
                        help='the rank of the machine as the source for snapshot')
    parser.add_argument('--dst', '-d', type=int, default=1,
                        help='the rank of the machine as the destination for snapshot')
    args = parser.parse_args()
    recover_snapshot = RecoverSnapshot(args)
    recover_snapshot.recover_GPU_snapshot()