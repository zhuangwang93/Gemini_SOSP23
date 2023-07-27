import os
import subprocess
import time
import argparse
from deepspeed.utils import logger
from boto3_agent import ASGAgent
from helper import sync_code, setup_instances
from deepspeed.runtime.snapshot.helper import perror, thread, etcdKey
from deepspeed.runtime.snapshot.etcd_client import etcdAPIs
    

class TraingLaunch():
    def __init__(self, args):
        self.instances_num = args.instances
        self.gpus_per_instance = args.gpus_per_instance
        self.example_dir = args.example_dir
        self.snapshot_dir = args.snapshot_dir
        self.ASG_agent = ASGAgent()
        self.ips_list = self.update_ips_list()
        self.etcd_port = 2379


    def update_ips_list(self):
        return self.ASG_agent.get_instances_ips(instance_state="running")


    def get_ips_list(self):
        return self.ips_list


    def generate_hostfile(self, filename="hostfile"):
        filename = os.path.join(self.example_dir, "hostfile")
        print(self.ips_list)
        with open(filename, 'w') as f:
            for ip in self.ips_list:
                f.write(f"{ip} slots={self.gpus_per_instance}\n")


    def run_etcd(self, num):
        self.set_etcd_hosts(num)
        # TODO: detect if etcd has been running
        self.start_etcd(num)


    def run_etcd_from_ips(self, ips):
        self.start_etcd_from_ips(ips)


    def set_etcd_hosts(self, num):
        if num <= self.instances_num:
            self.etcd_host_ips = self.ips_list[:num]
        else:
            self.etcd_host_ips = self.ips_list
            num = self.instances_num


    def start_etcd(self, num):
        REGISTRY = "gcr.io/etcd-development/etcd"
        ETCD_VERSION = "latest"
        TOKEN = "my-etcd-token"
        CLUSTER_STATE="new"
        names = ["etcd-node-"+str(i) for i in range(num)]
        CLUSTER = ""
        for name, host in zip(names, self.etcd_host_ips):
            CLUSTER += f"{name}=http://{host}:2380,"
        print(CLUSTER)
        DATA_DIR = "etcd-data"

        for name, host in zip(names, self.etcd_host_ips):
            THIS_NAME = name
            THIS_IP = host

            etcd_command = f"""
                docker volume create --name {DATA_DIR}; \
                docker run --rm -d -p 2379:2379 -p 2380:2380 -v {DATA_DIR}:/etcd-data \
                    --name etcd_container {REGISTRY}:{ETCD_VERSION} \
                    etcd --data-dir=/etcd-data --name {THIS_NAME} \
                    --initial-advertise-peer-urls http://{THIS_IP}:2380 --listen-peer-urls http://0.0.0.0:2380 \
                    --advertise-client-urls http://{THIS_IP}:{self.etcd_port} --listen-client-urls http://0.0.0.0:{self.etcd_port} \
                    --initial-cluster {CLUSTER} \
                    --initial-cluster-state {CLUSTER_STATE} --initial-cluster-token {TOKEN}
            """
            print(etcd_command)
            run_etcd_command = f"ssh -T {host} '{etcd_command}'"
            subprocess.Popen(run_etcd_command, shell=True)
            logger.info("***************************")
            logger.info(f"Start etcd on host {THIS_IP}...")


    def start_etcd_from_ips(self, ips):
        REGISTRY = "gcr.io/etcd-development/etcd"
        ETCD_VERSION = "latest"
        TOKEN = "my-etcd-token"
        CLUSTER_STATE="new"
        ips = ips.strip().split(" ")
        num = len(ips)
        names = ["etcd-node-"+str(i) for i in range(num)]
        CLUSTER = ""
        for name, host in zip(names, ips):
            CLUSTER += f"{name}=http://{host}:2380,"
        print(CLUSTER)
        DATA_DIR = "etcd-data"

        for name, host in zip(names, ips):
            THIS_NAME = name
            THIS_IP = host

            etcd_command = f"""
                docker volume create --name {DATA_DIR}; \
                docker run --rm -d -p 2379:2379 -p 2380:2380 -v {DATA_DIR}:/etcd-data \
                    --name etcd_container {REGISTRY}:{ETCD_VERSION} \
                    etcd --data-dir=/etcd-data --name {THIS_NAME} \
                    --initial-advertise-peer-urls http://{THIS_IP}:2380 --listen-peer-urls http://0.0.0.0:2380 \
                    --advertise-client-urls http://{THIS_IP}:{self.etcd_port} --listen-client-urls http://0.0.0.0:{self.etcd_port} \
                    --initial-cluster {CLUSTER} \
                    --initial-cluster-state {CLUSTER_STATE} --initial-cluster-token {TOKEN}
            """
            print(etcd_command)
            # run_etcd_command = f"ssh -T {host} '{etcd_command}'"
            # subprocess.Popen(run_etcd_command, shell=True)
            logger.info("***************************")
            logger.info(f"Start etcd on host {THIS_IP}...")
        


    def add_etcd(self):
        # TODO: https://etcd.io/docs/v3.5/tutorials/how-to-deal-with-membership/
        return


    def get_etcd_hosts(self):
        etcd_hosts = []
        for etcd_host in self.etcd_host_ips:
            etcd_hosts.append((etcd_host, self.etcd_port))
        self.etcd_hosts = tuple(etcd_hosts)
        etcd_hosts_info = ""
        for ip, port in self.etcd_hosts:
            etcd_hosts_info += f"{ip}:{port},"
        return etcd_hosts_info.strip(" ,")


    def start_agent(self):
        # reset etcd
        etcd_hosts_info = self.get_etcd_hosts()
        self.etcd = etcdAPIs(self.etcd_hosts)
        self.etcd.delete(etcdKey.root_agent_heartbeat_key)
        self.etcd.delete_folder(etcdKey.agent_heartbeat_folder)
        self.etcd.delete_folder(etcdKey.vote_folder)
        self.etcd.delete_folder(etcdKey.agent_ips_folder)

        path = "~/zhuang/failure_recovery/deepspeed/runtime/snapshot"
        for rank, ip in enumerate(self.ips_list):
            agent_command = f"""
                cd {path}; taskset -c 40 python3 agent_etcd.py -r {rank} \
                --ranks {self.instances_num} \
                --gpus-per-rank {self.gpus_per_instance} \
                --etcd-hosts {etcd_hosts_info} \
                --init-mode \
                --example-dir {self.example_dir} \
                --snapshot-dir {self.snapshot_dir} \
            """
            run_agent_command = f"ssh -T -v {ip} '{agent_command}'"
            subprocess.Popen(run_agent_command, shell=True)
            logger.info("***************************")
            logger.info(f"Start agent on host {ip} with instance rank {rank}...")


    def warmup_agent(self):
        for rank, ip in enumerate(self.ips_list):
            path = "~/zhuang/DeepSpeedExamples/bing_bert"
            agent_command = f"""
                cd {path}; bash deepspeed_bert_train_warmup.sh
            """
            run_agent_command = f"ssh -T {ip} '{agent_command}'"
            subprocess.Popen(run_agent_command, shell=True)
            logger.info("***************************")
            logger.info(f"Start agent on host {ip} with instance rank {rank}...")
        time.sleep(120)


    def kill_agent(self):
        for ip in reversed(self.ips_list):
            agent_command = f"""
                sudo pkill -f -9 python3; 
            """
            run_agent_command = f"ssh -T -v {ip} '{agent_command}'"
            subprocess.Popen(run_agent_command, shell=True)
            logger.info("***************************")
            logger.info(f"Kill agent on host {ip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='launch for fast snapshot',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode', '-m', type=str, default="none",
                        help='mode for training')
    parser.add_argument('--instances', '-i', type=int, default=2,
                        help='number of instances')
    parser.add_argument('--etcd-clients', '-c', type=int, default=1,
                        help='number of etcd clients')
    parser.add_argument('--etcd-ips', '-p', type=str, 
                        help='the ips of etcd clients')
    parser.add_argument('--gpus_per_instance', '-g', type=int, default=8,
                        help='number of GPUs in each instance')
    parser.add_argument('--example-dir', '-e', type=str, default=os.path.expanduser('~/zhuang/Gemini/examples'),
                        help='directory of deepspeed examples')
    parser.add_argument('--snapshot-dir', '-s', type=str, default=os.path.expanduser('/tmp/ramdisk/snapshot'),
                        help='directory of deepspeed snapshot')
    args = parser.parse_args()
    
    launch_agent = TraingLaunch(args)
    if args.mode == "init":
        sync_code(launch_agent.get_ips_list())
        setup_instances(launch_agent.get_ips_list())
    elif args.mode == "sync":
        sync_code(launch_agent.get_ips_list(), mode = "sync")
    elif args.mode == "etcd":
        launch_agent.run_etcd(args.etcd_clients)
    elif args.mode == "etcd_ip":
        launch_agent.run_etcd_from_ips(args.etcd_ips)
    elif args.mode == "start":
        launch_agent.generate_hostfile()
        launch_agent.set_etcd_hosts(args.etcd_clients)
        launch_agent.start_agent()
    elif args.mode == "kill":
        launch_agent.kill_agent()
    elif args.mode == "instances":
        launch_agent.generate_hostfile()
    else:
        print("set the mode from ['init', 'sync', 'etcd', 'start', 'kill', 'instances', 'etcd_ip']")