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



def sync_code(ips_list):
    for ip in ips_list:
        gemini_dir = "~/zhuang/Gemini"
        gemini_github = "https://github.com/zhuangwang93/Gemini.git"
        if not os.path.exists(gemini_dir):
            os.makedirs(gemini_dir)
            sync_code_command = f"""
                cd {gemini_dir}; git clone {gemini_github}
            """
        else:
            sync_code_command = f"""
                cd {gemini_dir}; git pull {gemini_github}
            """
        print(f"********{ip} sync code********")
        run_sync_command = f"ssh -T {ip} '{sync_code_command}'"
        subprocess.call(run_sync_command, shell=True)


def setup_instances(ips_list):
    for ip in ips_list:
        setup_command = f"""
            pip3 install transformers -U; mkdir /tmp/ramdisk; sudo mount -t tmpfs -o size=128G ramdisk /tmp/ramdisk; mount | tail -n 1
        """
        print(f"********{ip} update dependency********")
        run_update_command = f"ssh -T {ip} '{setup_command}'"
        subprocess.call(run_update_command, shell=True)