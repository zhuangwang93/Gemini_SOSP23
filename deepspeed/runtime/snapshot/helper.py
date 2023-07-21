import json
from threading import Thread
import socket
import sys
import os
import subprocess


def thread(fn):
    """To use as decorator to make a function call threaded.
    Needs import
    from threading import Thread"""
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper


def perror(msg):
    """Print a message to stderr"""
    sys.stderr.write(msg + '\n')


class ErrorCode():
    FATAL_ERROR_CODE = 3
    RECOVERABLE_ERROR_CODE = 2
    OTHER_ERROR_CODE = 1
    NORMAL_CODE = 0



class etcdKey():
    instance_ips_key = "/instance/ips"
    agent_state_key = "/agent/signals"
    agent_heartbeat_folder = "/agent/heartbeat"
    agent_ack_folder = "/agent/acks"
    root_agent_heartbeat_key = "/root/heartbeat"
    root_agent_rank_key = "/root/rank"
    root_agent_counter_key = "/root/counter"
    vote_folder = "/vote"
    agent_ips_folder = "/agent/ip"


def get_ip_address(ifname):
    import socket, fcntl, struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', bytes(ifname[:15], 'utf-8'))
    )[20:24])
    

def sync_code(ips_list, mode = "init"):
    local_ip = get_ip_address("eth0")
    gemini_parent_dir = os.path.join(os.path.expanduser('~'), "zhuang")
    gemini_github = "https://github.com/zhuangwang93/Gemini.git"

    for ip in ips_list:
        if ip == local_ip:
            continue
        if mode == "init":
            sync_code_command = f"cd {gemini_parent_dir}; git clone {gemini_github}; cd Gemini; pip3 install -e ."
        elif mode == "sync":
            gemini_dir = os.path.join(gemini_parent_dir, "Gemini")
            sync_code_command = f"cd {gemini_dir}; git pull {gemini_github}"
        
        print(f"********{ip} sync code********")
        print(sync_code_command)
        run_sync_command = f"ssh -T {ip} '{sync_code_command}'"
        subprocess.call(run_sync_command, shell=True)


def setup_instances(ips_list):
    gemini_dir = os.path.join(os.path.expanduser('~'), "zhuang/Gemini")
    for ip in ips_list:
        setup_command = f"""
            cd {gemini_dir}; pip3 install -e .; 
            pip3 install transformers -U; 
            mkdir /tmp/ramdisk; 
            sudo mount -t tmpfs -o size=128G ramdisk /tmp/ramdisk; mount | tail -n 1
        """
        print(f"********{ip} update dependency********")
        run_update_command = f"ssh -T {ip} '{setup_command}'"
        subprocess.call(run_update_command, shell=True)