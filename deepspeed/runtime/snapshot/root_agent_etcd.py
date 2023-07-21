#!/usr/bin/env python

"""Simple server using epoll."""

from __future__ import print_function
import time
import sys
import argparse
import os
import subprocess
from deepspeed.runtime.snapshot.helper import thread, etcdKey, ErrorCode, sync_code, setup_instances
from deepspeed.runtime.snapshot.etcd_client import etcdAPIs
from deepspeed.runtime.snapshot.snapshot_policy import GroupPolicy
from deepspeed.runtime.snapshot.boto3_agent import ASGAgent
from deepspeed.utils import logger


class RootAgentHeartbeat():
    def __init__(self, etcd, timer=5):
        self.timer = timer
        self.etcd = etcd
        self.heartbeat_flag = True
        self.heartbeat_counter = 0
        self.heartbeat_key = etcdKey.root_agent_heartbeat_key
        self.reset_heartbeat_counter()
        self.heartbeat_deamon()


    def reset_heartbeat_counter(self):
        message = {}
        message["counter"] = 0
        self.etcd.write(self.heartbeat_key, message)


    def construct_heartbeat_message(self):
        message = {}
        self.heartbeat_counter += 1
        message["counter"] = self.heartbeat_counter
        return message


    @thread
    def heartbeat_deamon(self):
        while self.heartbeat_flag:
            message = self.construct_heartbeat_message()
            self.etcd.write(self.heartbeat_key, message)
            time.sleep(self.timer)



class RootAgent():
    def __init__(self, args):
        self.role = "root_agent"
        # GPU machine ranks
        self.rank = args.rank
        # the number of GPU machines
        self.ranks = args.ranks
        # the number of GPUs on each machine
        self.gpus_per_rank = args.gpus_per_rank
        self.example_dir = args.example_dir
        self.snapshot_dir = args.snapshot_dir
        self.agent_heartbeat_counters = [-1] * self.ranks
        self.snapshot_version = 0
        self.timer = args.timer
        self.max_faulty_acks_cnt = 2
        self.faulty_acks_cnt = 0
        self.max_faulty_heartbeat_cnt = 3
        self.faulty_heartbeat_cnt = 0
        self.init_mode = args.init_mode
        self.failed_machines = []
        self.latest_ckpt_filenames = {}
        self.etcd_hosts_info = args.etcd_hosts
        self.wait_for_warmup = True
        self.init_etcd()
        self.ASG_agent = ASGAgent()
        self.heartbeats = RootAgentHeartbeat(self.etcd, 3)
        self.init_phase()
        self.agent_deamon()
        

    def set_hostfile(self):
        self.ips_list = self.ASG_agent.get_instances_ips()
        filename = os.path.join(self.example_dir, "hostfile")
        logger.info(f"hostfile filename: {filename}")
        self.ASG_agent.write_hostfile(filename, self.ips_list, self.gpus_per_rank)
        self.etcd.write(self.instance_ips_key, self.ips_list)
        

    def extract_etcd_hosts(self):
        etcd_hosts = self.etcd_hosts_info.strip(" ,").split(",")
        hosts = []
        for host in etcd_hosts:
            values = host.split(":")
            ip, port = values[0], int(values[1])
            hosts.append((ip, port))
        return tuple(hosts)

    
    def init_etcd(self):
        self.etcd_hosts = self.extract_etcd_hosts()
        self.etcd = etcdAPIs(self.etcd_hosts)
        # TODO: is instance_ips_key a must?
        self.instance_ips_key = etcdKey.instance_ips_key
        self.agent_heartbeat_folder = etcdKey.agent_heartbeat_folder
        self.root_agent_heartbeat_key = etcdKey.root_agent_heartbeat_key
        self.agent_ack_folder = etcdKey.agent_ack_folder
        self.root_agent_rank_key = etcdKey.root_agent_rank_key
        self.vote_folder = etcdKey.vote_folder
        self.agent_ip_folder = etcdKey.agent_ips_folder


    def init_phase(self):
        self.snapshot_groups = self.construct_snapshot_group()
        self.phase = "init"
        self.faulty_heartbeat_cnt = 0
        self.etcd.write(self.root_agent_rank_key, self.construct_root_agent_message())
        # there are three cases:
        # 1. the begining of the training. the root agent needs to start the training
        # 2. some machines fail. The root agent needs to recover training from failure
        # 3. only the root agent fails and the training is still there. Nothing to do
        if self.init_mode:
            self.set_hostfile()
            self.start_training()
        else:
            self.etcd.delete(etcdKey.root_agent_heartbeat_key)
            self.failed_machines = self.get_failed_machines_ranks()
            if len(self.failed_machines) > 0:
                logger.info(f"Recover failure, failed machines: {self.failed_machines}")
                self.failure_recovery()
            else:
                logger.info(f"All agents are alive.")
        
        # TODO: what if training fails but agents are still alive?
        time.sleep(self.timer * 2)
        self.etcd.delete_folder(self.vote_folder)
        self.init_mode = False
    

    def construct_root_agent_message(self): 
        message = {}
        message["rank"] = self.rank
        return message


    def get_failure_type(self):
        return ErrorCode.OTHER_ERROR_CODE


    def failure_recovery(self, error_type=ErrorCode.OTHER_ERROR_CODE):
        if self.phase != "recovery":
            self.phase = "recovery"
            self.wait_for_warmup = True
            self.kill_training()
            if error_type == ErrorCode.FATAL_ERROR_CODE:
                self.replace_instances()
            elif error_type == ErrorCode.RECOVERABLE_ERROR_CODE:
                self.reboot_instances()
            else:
                self.restart_agent()
            self.restart_training()


    def reboot_instances(self):
        # TODO
        self.replace_instances()


    def get_failed_machines_ranks(self):
        votes = self.etcd.read_folder(self.vote_folder)
        alive_machines = {int(rank) for rank in votes.keys()}
        all_machines = set(range(self.ranks))
        failed_machines = all_machines.difference(alive_machines)
        return list(failed_machines)


    def recv_signals_from_etcd(self, key):
        signals = self.etcd.read_folder(key)
        return {int(key) : value for key, value in signals.items()}


    def add_failed_machine(self, rank):
        if rank not in self.failed_machines:
            self.failed_machines.append(rank)


    def has_failed_machines(self):
        return len(self.failed_machines) > 0


    def recv_heartbeat_signal(self):
        while True:
            signals = self.recv_signals_from_etcd(self.agent_heartbeat_folder)
            if len(signals) < self.ranks:
                time.sleep(self.timer)
            else:
                break

        self.failed_machines = []
        for rank in range(self.ranks):
            message = signals[rank]
            if message["counter"] <= self.agent_heartbeat_counters[rank]:
                self.add_failed_machine(rank)
            self.agent_heartbeat_counters[rank] = message["counter"]
        
        if self.has_failed_machines():
            if self.wait_for_warmup:
                return True
            logger.debug(f"Heartbeat received from all agents: {signals}")
            self.faulty_heartbeat_cnt += 1
            if self.faulty_heartbeat_cnt == self.max_faulty_heartbeat_cnt:
                self.faulty_heartbeat_cnt = 0
                logger.info("***************************")
                logger.info(f"Heartbeat not receive from machines {self.failed_machines}")
                self.failure_recovery()     
        else:
            if self.wait_for_warmup:
                self.wait_for_warmup = False
                logger.info("Warmup for failure recovery ended...")
            self.faulty_heartbeat_cnt = 0
        return True


    def run_deepspeed(self, bash_file):
        run_deepspeed_command = f"cd {self.example_dir}; bash {bash_file}"
        return subprocess.Popen(run_deepspeed_command, shell=True)

    
    def start_training(self):
        if self.phase != "start":
            self.phase = "start"
            bash_file = "launch.sh"
            logger.info(f"Start training...")
            self.run_deepspeed(bash_file)
            

    # debug: delete existing checkpoints before starting training
    def restart_training(self):
        bash_file = "relaunch.sh"
        logger.info("***************************")
        logger.info(f"Restart training from checkpoint...")
        self.run_deepspeed(bash_file)
        self.phase = "snapshot"


    def kill_training(self):
        # TODO: how to gracefully kill training process?
        kill_deepspeed_command = "pkill -f -9 deepspeed_train"
        subprocess.call(kill_deepspeed_command, shell=True)
        logger.info(f"Killed training")
    

    def recover_snapshot(self, ifname="eth0"):
        def write_hostfile(ips, slots, filename):
            dir = "rankfiles"
            if not os.path.exists(dir):
                os.makedirs(dir)
            rankfile = os.path.join(dir, filename)
            rank = 0
            with open(rankfile, 'w') as f:
                for ip in ips:
                    for i in range(slots):
                        f.write(f"rank {rank}={ip} slots={i + 1}\n")
                        rank += 1
            return rankfile

        logger.info("***************************")
        logger.info(f"Machines for recover: {self.failed_machines}, ip list: {self.ips_list}")

        pairs = self.get_snapshot_recovery_pair()
        if pairs is None:
            self.recover_from_storage()
        else:
            logger.info(f"Recovery pairs: {pairs}")
            mpi_config = f"""mpirun --allow-run-as-root --mca oob_tcp_if_include {ifname} \
                -mca btl_tcp_if_include {ifname} -x NCCL_SOCKET_IFNAME={ifname} \
                -bind-to none -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH \
                -x PATH -mca pml ob1 -mca btl ^openib """
            
            cnt = 0
            for pair in pairs:
                src, dst = pair
                src_ip, dst_ip = self.ips_list[src], self.ips_list[dst]
                rankfile = f"rankfile_{cnt}"
                cnt += 1
                rankfile = write_hostfile([src_ip, dst_ip], self.gpus_per_rank, rankfile)
                command = mpi_config + f"-np {self.gpus_per_rank*2} -H {src_ip},{dst_ip} -rf {rankfile} python3.7 snapshot_recover.py --src {src} --dst {dst}"
                logger.info(f"{command}")
                subprocess.call(command, shell=True)
                logger.info(f"Recover snapshot {src} -> {dst}")
            

    # TODO:
    def recover_from_storage(self):
        return True


    def construct_snapshot_group(self, group_size=2):
        return GroupPolicy(self.ranks, self.gpus_per_rank, group_size=group_size)
    

    def get_snapshot_machine_group(self, rank):
        return self.snapshot_groups.get_machine_group(rank)


    def restart_agent(self):
        ips = self.etcd.read_folder(self.agent_ip_folder, dict_type=False)
        for rank in self.failed_machines:
            ip = ips[str(rank)]
            path = "~/zhuang/failure_recovery/deepspeed/runtime/snapshot"
            agent_command = f"""
                cd {path}; \
                taskset -c 40 python3.7 agent_etcd.py \
                -r {rank} \
                --ranks {self.ranks} \
                --gpus-per-rank {self.gpus_per_rank} \
                --etcd-hosts {self.etcd_hosts_info} \
                --root-alive \
                --example-dir {self.example_dir} \
                --snapshot-dir {self.snapshot_dir} \
            """
            run_agent_command = f"ssh -T -v {ip} '{agent_command}'"
            subprocess.Popen(run_agent_command, shell=True)
            logger.info("***************************")
            logger.info(f"Restart agent on instance {ip} with rank {rank}...")



    def replace_instances(self):
        # Step 1: terminate failed instances
        # Step 2: rely on ASG to replace instances
        # Step 3: recover snapshot to newly added instances
        # Step 4: start agents on newly added instances
        # Step 5: update hostfile for training
        instances_info = self.ASG_agent.get_instances_info()
        pre_ips_list = self.ASG_agent.get_instances_ips(instances_info)
        # When instances crash, is it listed in instances_info?
        ips_list = self.etcd.read(self.instance_ips_key)
        assert(len(instances_info) == self.ranks)
        ips = self.etcd.read_folder(self.agent_ip_folder, dict_type=False)

        for rank in self.failed_machines:
            ip = ips[str(rank)]
            instance_id = instances_info[ip][1]
            logger.info(f"{rank}, {ip}, {instance_id}")
            self.ASG_agent.terminate_instance(instance_id)
            logger.info(f"[replace instances] terminate instance {instance_id} with IP {ip}")

        # wait for a while when ASG adds new instances until all instances are ready
        time.sleep(30)
        while True:
            new_instances_info = self.ASG_agent.get_instances_info(instance_state="running")
            if len(new_instances_info) == self.ranks:
                self.ips_list = self.ASG_agent.get_instances_ips(new_instances_info)
                break
            time.sleep(5)
        time.sleep(20)
        
        # reset heartbeat from agents
        self.etcd.delete_folder(etcdKey.agent_heartbeat_folder)
        # detect the new added instances
        print(f"previous IP list: {pre_ips_list}, new IP list: {self.ips_list}")
        self.ips_list, new_ips, del_ips, index_list = self.ASG_agent.diff_ips(pre_ips_list, self.ips_list)
        assert(len(self.failed_machines) == len(new_ips))
        logger.info(f"[replace instances] ip list: {self.ips_list}, new ips: {new_ips}, del ips: {del_ips}")
        
        time.sleep(40)
        for ip, rank in zip(new_ips, index_list):
            subprocess.call(f"ssh -T -v {ip} 'echo Hello'", shell=True)
            logger.info(f"{ip} sync code")
            sync_code([ip])
            setup_instances([ip])
            # start agents on newly added instances
            path = "~/zhuang/failure_recovery/deepspeed/runtime/snapshot"
            agent_command = f"""
                cd {path}; \
                taskset -c 40 python3.7 agent_etcd.py 
                -r {rank} \
                --ranks {self.ranks} \
                --gpus-per-rank {self.gpus_per_rank} \
                --etcd-hosts {self.etcd_hosts_info} \
                --root-alive \
                --example-dir {self.example_dir} \
                --snapshot-dir {self.snapshot_dir} \
            """
            run_agent_command = f"ssh -T -v {ip} '{agent_command}'"
            subprocess.Popen(run_agent_command, shell=True)
            logger.info("***************************")
            logger.info(f"Start agent on instance {ip} with rank {rank}...")
            
        self.recover_snapshot()
        # update hostfile for training
        filename = os.path.join(self.example_dir, "hostfile")
        self.ASG_agent.generate_hostfile(filename, self.ips_list, self.gpus_per_rank)
        self.etcd.write(self.instance_ips_key, self.ips_list)

        

    def get_ranks_for_snapshot(self, rank):
        return self.get_snapshot_machine_group(rank)


    def get_snapshot_recovery_pair(self):
        pairs = []
        for rank in self.failed_machines:
            # reset the heartbeat counters of the failed machines
            self.agent_heartbeat_counters[rank] = -1
            ranks = self.get_ranks_for_snapshot(rank)
            rank_recover_from_snapshot = False
            for r in ranks:
                if r not in self.failed_machines:
                    pairs.append((r, rank))
                    rank_recover_from_snapshot = True
                    break
            if not rank_recover_from_snapshot: 
                logger.info(f"Cannot recover failures from snapshot. Recover from the storage system.")
                return None
        return pairs



    @thread
    def agent_deamon(self):
        time.sleep(20)
        while True:
            self.recv_heartbeat_signal()
            time.sleep(self.timer)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manager for fast snapshot',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--rank', '-r', type=int,
                        help='machine rank')
    parser.add_argument('--ranks', type=int,
                        help='number of ranks in training')
    parser.add_argument('--gpus-per-rank', type=int, default=8,
                        help='number of GPU machines in training')
    parser.add_argument('--timer', '-t', type=int, default=5,
                        help='snapshot timer')
    parser.add_argument('--example-dir', '-e', type=str, default=os.path.expanduser('~/zhuang/DeepSpeedExamples/bing_bert'),
                        help='directory of deepspeed examples')
    parser.add_argument('--snapshot-dir', '-s', type=str, default=os.path.expanduser('/tmp/ramdisk/snapshot'),
                        help='directory of deepspeed snapshot')
    parser.add_argument('--init-mode', action='store_true', default=False,
                        help='init root agent')
    parser.add_argument('--etcd-hosts', type=str,
                        help='the ips and ports of etcd members')   
    args = parser.parse_args()
    try:
        manager = RootAgent(args)
    except KeyboardInterrupt as e:
        logger.error("The Root Agent is shutdown")