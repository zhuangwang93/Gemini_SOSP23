[![License MIT](https://badgen.net/badge/license/MIT/blue)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/deepspeed.svg)](https://pypi.org/project/deepspeed/)
[![Downloads](https://pepy.tech/badge/deepspeed)](https://pepy.tech/project/deepspeed)
[![Build](https://badgen.net/badge/build/check-status/blue)](#build-pipeline-status)


# Artifact Evalution of GEMINI

This repository contains the system code and scripts that help run the GEMINI experiments from our SOSP '23 paper.


## Prerequisites

- DeepSpeed == 0.73
- CUDA == 11.6
- PyTorch >= 1.13.0
- NCCL >= 2.14.3
- etcd == 3.5

## Machines

All the experiments in the main body of GEMINI paper are conducted on 16 AWS p4d.24xlarge instances with 128 A100 GPUs. The number of parameters in the evaluated models are 100 billion. The network bandwidth connecting machines is 400Gbps.
However, we don't expect that the AEC members can have access to such a large number of GPUs to fit these 100B models.
Therefore, we aims at minimal working examples with all models of 5 billion parameters in this AE.
AEC members can use 32 V100 GPUs in 4 AWS p3dn.24xlarge instances with auto scaling group (ASG) for the model training.
If AEC members still think it is difficult to use the scale of GPU hardware for the evaluations, please contact us and we can reserve 4 AWS p3dn.24xlarge instances for AE in our GPU cluster.


## Installation

Install GEMINI on each machine before running any experiments. Please make sure the machines can successfully install [DeepSpeed](https://github.com/microsoft/DeepSpeed).
If AEC members directly use our machines for the evalution, we will have GEMINI and all dependencies pre-installed on all machines.

```bash
git clone https://github.com/zhuangwang93/Gemini.git
cd Gemini
pip3 install -e .
pip3 install transformers -U 
```

## Code structure

Gemini is built upon DeepSpeed and its main code is under [snapshot](deepspeed/runtime/snapshot/). The examples to run Gemini are under [examples](examples/).


## How to run

You can follow the instructions for evaluations if you'd like run GEMINI on your machines

```bash
# Step 1: replace the ip addresses of the machines in examples/hostfile. 
# If you are using ASG for the instances, you can also automatically set the ip addresses with
cd deepspeed/runtime/snapshot
python3 launch -m instances

# Step 2: start etcd as the distributed key-value store
cd deepspeed/runtime/snapshot
# If you are using ASG for the instances, you can start etcd with
python3 launch -m etcd
# Otherwise
python3 launch -m etcd_ip --etcd-ips "IP1 IP2 ..."

# Step 3: run the experiments. Note that we provide a script to run all experiments with one command in the next section.
cd deepspeed/examples/model_name
bash launch.sh
```


## Artifact Evaluation

Our claim of this paper is: GEMINI can checkpoint the model states for every iteration and it incurs negligible overhead on the training throughput.
The figures that can demonstrate this claim is Figure 7 (iteration time) and Figure 8 (the network idle time). 


```bash
# For Figure 7 and Figure 9. It may take around 30 min. The raw data, including both the iteration time of each step and the network idle time, is stored in results.json.
# The first 8 step time is for warm up; the middle 10 step time is without any check; the remaining step time is with GEMINI. 
cd deepspeed/examples/SOSP_AE
bash run_all.sh
# After running the script, Figure_7 and Figure_8 will appear in the folder. Because of the different experimental settings, the absolute values in these figures may vary from those provided in the paper. But you will see that the iteration time is almost the same without checkpoint and with GEMINI.

# The data collected in Figure 9, Figure 11, Figure 13, and Figure 14 are from simulations. We also provide the simulation code in AE_figures.ipynb. 
# You can play with them if interested.
```