#!/usr/bin/env python

"""Simple server using epoll."""

from __future__ import print_function
import time
import sys
import os
import argparse
import subprocess
from deepspeed.runtime.snapshot.helper import perror, thread, etcdKey
from deepspeed.runtime.snapshot.etcd_client import etcdAPIs
from deepspeed.runtime.snapshot.snapshot_record import SnapshotRecord
from deepspeed.runtime.snapshot.GPU_health_check import HealthCheck
from deepspeed.utils import logger


class AgentHeartbeat():
    def __init__(self, args, etcd, timer, keyword):
        self.rank = args.rank
        self.timer = timer
        self.keyword = keyword
        self.etcd = etcd
        self.heartbeat_flag = True
        self.heartbeat_counter = 0
        self.agent_heartbeat_key = f"{etcdKey.agent_heartbeat_folder}/{self.rank}"
        self.init_health_check()


    def check_health_status(self, state):
        return self.health_check.check(state)


    def init_health_check(self):
        self.health_check = HealthCheck(self.keyword)
        self.check_gpu_health_deamon()
        
        
    def construct_heartbeat_message(self):
        message = {}
        self.heartbeat_counter += 1
        message["counter"] = self.heartbeat_counter
        state = "normal" if self.heartbeat_counter > 5 else "init"
        message["status"] = self.check_health_status(state=state)
        return message


    @thread
    def check_gpu_health_deamon(self):
        while self.heartbeat_flag:
            message = self.construct_heartbeat_message()
            self.etcd.write(self.agent_heartbeat_key, message)
            logger.info(f"[agent {self.rank} heartbeat]: {message}")
            time.sleep(self.timer)


class Agent():
    def __init__(self, args, keyword="deepspeed"):
        self.role = "agent"
        self.rank = args.rank
        self.ranks = args.ranks
        self.gpus_per_rank = args.gpus_per_rank
        self.example_dir = args.example_dir
        self.snapshot_dir = args.snapshot_dir
        self.snapshot_record = SnapshotRecord(self.snapshot_dir, self.rank * self.gpus_per_rank)
        self.last_record = self.snapshot_record.read()
        self.agent_flag = True
        self.keyword = keyword
        self.phase = "init"
        self.max_faulty_root_heartbeat_cnt = 200
        self.faulty_root_heartbeat_cnt = 0
        self.timer = args.timer
        self.init_mode = args.init_mode
        self.root_alive = args.root_alive
        self.etcd_hosts_info = args.etcd_hosts
        self.etcd_hosts = self.extract_etcd_hosts()
        self.wait_for_warmup = True
        self.init_etcd()
        self.heartbeat = AgentHeartbeat(args, self.etcd, timer=3, keyword="deepspeed")
        self.root_agent_heartbeat_counter = -1
        self.root_agent_message_id = -1
        self.agent_deamon()
        

    def extract_etcd_hosts(self):
        etcd_hosts = self.etcd_hosts_info.strip(" ,").split(",")
        hosts = []
        for host in etcd_hosts:
            values = host.split(":")
            ip, port = values[0], int(values[1])
            hosts.append((ip, port))
        return tuple(hosts)


    def init_etcd(self):
        self.etcd = etcdAPIs(self.etcd_hosts)
        self.agent_heartbeat_folder = etcdKey.agent_heartbeat_folder
        self.root_agent_heartbeat_key = etcdKey.root_agent_heartbeat_key
        self.root_agent_rank_key = etcdKey.root_agent_rank_key
        self.vote_folder = etcdKey.vote_folder
        self.vote_key = f"{self.vote_folder}/{self.rank}"
        self.agent_ip_folder = etcdKey.agent_ips_folder
        self.agent_ip_key = f"{self.agent_ip_folder}/{self.rank}"

        self.write_agent_ip()
        if not self.root_alive:
            self.vote_for_leader()
        else:
            message = self.etcd.read(self.root_agent_rank_key)
            self.root_agent_rank = message["rank"]


    def write_agent_ip(self, ifname="eth0"):
        def get_ip_address(ifname):
            import socket, fcntl, struct
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            return socket.inet_ntoa(fcntl.ioctl(
                s.fileno(),
                0x8915,  # SIOCGIFADDR
                struct.pack('256s', bytes(ifname[:15], 'utf-8'))
            )[20:24])

        local_ip = get_ip_address(ifname) 
        self.etcd.write(self.agent_ip_key, local_ip)


    def vote_for_leader(self):
        self.etcd.write(self.vote_key, self.rank)
        # wait for all alive agents to hand up
        time.sleep(self.timer)
        start_training_flag = False
        if self.init_mode:
            self.root_agent_rank = self.wait_for_all_agents()
            self.init_mode = False
            start_training_flag = True
        else:
            start_training_flag = False
            self.root_agent_rank, _ = self.select_leader()  

        self.phase = "train"
        self.root_agent_heartbeat_counter = -1
        self.root_agent_message_id = -1
        logger.info("\n***************************")
        logger.info(f"[Agent {self.rank}] Root agent is Rank {self.root_agent_rank}")
        if self.root_agent_rank == self.rank:
            self.role = "root_agent"
            self.run_root_agent(start_training_flag)
            

    def wait_for_all_agents(self):
        while True:
            votes = self.etcd.read_folder(self.vote_folder, dict_type=False)
            if len(votes) == self.ranks:
                return 0
            else:
                time.sleep(5)
                logger.info(f"{votes} are ready. wait for all {self.ranks} agents to get ready...")


    def select_leader(self):
        alive_ranks = []
        votes = self.etcd.read_folder(self.vote_folder, dict_type=False)
        logger.info(f"[select leader] votes: {votes}")
        for rank in votes.keys():
            alive_ranks.append(int(rank))
        return min(alive_ranks), len(alive_ranks)


    def run_root_agent(self, start_training):
        run_root_agent_command = f"""
            python3.7 root_agent_etcd.py -r {self.rank} \
            --ranks {self.ranks} \
            --gpus-per-rank {self.gpus_per_rank} \
            --etcd-hosts {self.etcd_hosts_info} \
            --example-dir {self.example_dir} \
            --snapshot-dir {self.snapshot_dir} \
            """
        if start_training:
            run_root_agent_command += " --init-mode "
        logger.info(run_root_agent_command)
        subprocess.Popen(run_root_agent_command, shell=True)
        logger.info(f"start root agent on rank {self.rank}...")
        return True


    def check_root_agent_heartbeat(self):
        signal = self.etcd.read(self.root_agent_heartbeat_key)
        counter = int(signal["counter"])
        if counter > self.root_agent_heartbeat_counter:
            self.root_agent_heartbeat_counter = counter
            self.faulty_root_heartbeat_cnt = 0
            if self.wait_for_warmup == True:
                self.wait_for_warmup = False
                self.max_faulty_root_heartbeat_cnt = 2
        else:
            if self.wait_for_warmup:
                return
            logger.info(f"[agent {self.rank}] check root agent heartbeat {signal}, self.root_agent_heartbeat_counter")
            # when the root agent and its corresponding agent fail at the same time (this is a common case):
            # Step 0. the alive agents kill their training programs
            # Step 1. the alive agents vote for the rank for the root agent
            # Step 2. the root agent reads the vote folder to figure out the alive agents and the failed machines
            # Step 3. the root agent issues the following states: replace -> recover -> restart -> snapshot
            self.faulty_root_heartbeat_cnt += 1
            if self.faulty_root_heartbeat_cnt == self.max_faulty_root_heartbeat_cnt:
                self.faulty_root_heartbeat_cnt = 0
                if self.phase != "vote":
                    self.phase = "vote"
                    logger.info(f"[Agent {self.rank}] Root agent {self.root_agent_rank} is down. Vote for a new leader")
                    self.vote_for_leader()
                


    @thread
    def agent_deamon(self):
        while self.agent_flag:
            self.check_root_agent_heartbeat()
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
                        help='init agent')
    parser.add_argument('--root-alive', '-a', action='store_true', default=False,
                        help='whether the root agent is alive')
    parser.add_argument('--etcd-hosts', type=str,
                        help='the ips and ports of etcd members')           
    args = parser.parse_args()
    try:
        manager = Agent(args)
    except KeyboardInterrupt as e:
        logger.error("The Agent is shutdown")