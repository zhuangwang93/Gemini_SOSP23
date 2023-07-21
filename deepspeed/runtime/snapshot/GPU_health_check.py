#!/usr/bin/env python3
import os

import requests
import torch
import sys
import subprocess
import time
from deepspeed.runtime.snapshot.helper import ErrorCode


BASE_URL = 'http://169.254.169.254/latest/meta-data/'
INSTANCE_ID = requests.get(BASE_URL + 'instance-id').text
INSTANCE_TYPE = requests.get(BASE_URL + 'instance-type').text
EC2_REGION = requests.get(BASE_URL + 'placement/availability-zone').text[:-1]


class HealthCheck():
    def __init__(self, keyword):
        self.instance_type = INSTANCE_TYPE
        self.keyword = keyword
        self.first_check()

    
    def first_check(self):
        num_gpus_map = {'g4dn.12xlarge':4, 'p3.2xlarge': 1, 'p3.8xlarge': 4, 'p3.16xlarge': 8, 'p3dn.24xlarge': 8, 'p4d.24xlarge': 8}
        assert(self.instance_type in num_gpus_map)
        self.num_gpus = num_gpus_map[self.instance_type]
        self.devices = [torch.device('cuda:' + str(i)) for i in range(self.num_gpus)]

        self.run_nvsmi_test()
        self.run_tensor_test()


    def run_tensor_test(self):
        for x in range(self.num_gpus):
            kwargs = {'dtype': torch.float32,
                'device': self.devices[x],
                'requires_grad': False}
            try:
                torch.ones(8, **kwargs)
            except RuntimeError:
                return ErrorCode.FATAL_ERROR_CODE
        return ErrorCode.NORMAL_CODE


    def run_nvsmi_test(self):
        """
        Check for ECC errors in nvidia-smi
        We are looking for the following error logs:
        Pending Page Blacklist : Yes
        Remapping Failure Occurred : Yes
        """
        try:
            out = subprocess.run("nvidia-smi", timeout=20, capture_output=True)
            if out.returncode < 0:
                return ErrorCode.FATAL_ERROR_CODE
        except subprocess.TimeoutExpired:
            return ErrorCode.FATAL_ERROR_CODE

        if self.nvsmi_check_occurence("'Pending Page Blacklist'"):
            return ErrorCode.RECOVERABLE_ERROR_CODE

        if self.nvsmi_check_occurence("'Remapping Failure Occurred'"):
            return ErrorCode.FATAL_ERROR_CODE
        return ErrorCode.NORMAL_CODE

        
    def run_cmd(self, cmd):
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = ps.communicate()[0]
        lines = output.decode().strip().split('\n')
        return lines


    def nvsmi_check_occurence(self, sentence):
        cmd = "nvidia-smi -q | grep " + sentence
        lines = self.run_cmd(cmd)
        for line in lines:
            if 'Yes' in line:
                return True
        return False


    def run_program_test(self):
        cmd = "ps aux | grep " + self.keyword
        lines = self.run_cmd(cmd)
        if len(lines) == self.num_gpus:
            return ErrorCode.NORMAL_CODE
        else:
            return ErrorCode.RECOVERABLE_ERROR_CODE


    def check(self, state="normal"):
        # subprocess.run(['dmesg', '-T'])
        tests = [self.run_nvsmi_test, self.run_tensor_test]
        error_code = ErrorCode.NORMAL_CODE
        for test in tests:
            error_code = max(error_code, test())

        if state == "normal":
            error_code = max(error_code, self.run_program_test())
        return error_code