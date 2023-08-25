[![License Apache 2.0](https://badgen.net/badge/license/apache2.0/blue)](https://github.com/Microsoft/DeepSpeed/blob/master/LICENSE)


# Artifact Evaluation of SOSP 2023 #30

This repository contains the system code and scripts that help run the experiments of our SOSP '23 paper (#30 in AE).


## Prerequisites

- DeepSpeed == 0.73
- CUDA == 11.6
- PyTorch >= 1.13.0
- NCCL >= 2.14.3
- etcd == 3.5
- Auto Scaling Group in AWS
- OS: Linux and other OS supported by DeepSpeed

## Machines

**Machines used in the paper:** All the experiments in the main body of our paper are conducted on 16 AWS p4d.24xlarge instances with 128 A100 GPUs. The number of parameters in the evaluated models is 100 billion.

**Minimal working example:** In this AE, we aim at minimal working examples with 32 V100 GPUs in 4 AWS p3dn.24xlarge instances with auto scaling group (ASG) for the model training.
The number of parameters in models is around 5 billion. A larger model size might cause out-of-memory issues and crash training.

**Use our machines:** We can provide 4 AWS p3dn.24xlarge instances for AE in our GPU cluster. Please contact us if needed.


## Installation

Install the code on each machine before running any experiments. Please make sure the machines can successfully install [DeepSpeed](https://github.com/microsoft/DeepSpeed).
If AEC members directly use our machines for the evaluation, we will have all dependencies pre-installed on all machines.

```bash
git clone https://github.com/zhuangwang93/SOSP-30_AE.git
cd SOSP-30_AE
pip3 install -e .
pip3 install transformers -U 
```

## Code Structure

**Main code:** The system is built upon DeepSpeed and its main code is under [snapshot](deepspeed/runtime/snapshot/). We also hack `deepspeed/runtime/zero/stage3.py` and `deepspeed/runtime/zero/partitioned_param_coordinator.py` to enable the checkpoint of optimizer states for every iteration.

**Examples:** The examples are under [examples](examples/). Three models are used for evaluation: GPT, BERT, and Roberta.

**AE scripts:** The scripts to run the artifact evaluation are under [SOSP_AE](examples/SOSP_AE).


## How to run

You can follow the instructions for evaluations if you'd like to run the code on your machines.

```bash
# Note: the path under our testbed is ~/zhuang/Gemini
# cd SOSP-30_AE
# Step 1: replace the IP addresses of the machines in examples/hostfile. 
# If you are using ASG for the instances, you can also automatically set the IP addresses with
cd deepspeed/runtime/snapshot
python3 launch.py -m instances

# Step 2: start etcd as the distributed key-value store
cd deepspeed/runtime/snapshot
# If you are using ASG for the instances (strongly recommended), you can start etcd with
python3 launch.py -m etcd
# Otherwise.
python3 launch.py -m etcd_ip --etcd-ips "IP1"

# Step 3: run the model script.
# Note that we also provide a script to run all experiments with one command in the next section.
# Note: the path under our testbed is ~/zhuang/Gemini/examples/model_name
cd examples/model_name
bash launch.sh
```


## Artifact Evaluation

### The main claim

**Our main claim:** `Our system can checkpoint the model states for every iteration and it incurs negligible overhead on the training throughput`.

The figures that can demonstrate this claim are Figure 7 (iteration time) and Figure 8 (the network idle time). 

```bash
# Note: the path under our testbed is ~/zhuang/Gemini/examples/SOSP_AE
cd examples/SOSP_AE
# It will take about 30 min to finish the experiments of the three models.
bash run_all.sh
```
The raw data, including both the iteration time of each step and the network idle time, is stored in `results.json`.
The first 8 step times are for warm-up; the middle 10 step times are without any checkpoints; the remaining step times are with GEMINI for checkpointing.
In addition, the generated checkpoints are stored under `model_name/snapshot/`. `x.pt` is the local checkpoint for `Rank x` and `x_y.pt` is the remote checkpoint for `Rank y` stored in `Rank x`.  

After running the script, Figure_7 and Figure_8 will appear in the folder. 
Because of the different experimental settings, the absolute values in these figures may vary from those provided in the paper. 
But you will see that the iteration time is almost the same without checkpoints and with GEMINI for checkpointing.
You can also set `--snapshot_mode` in launch.sh to `naive` from `interleave` to see how traffic blocking for checkpointing affects the iteration time.

### Ablation study

The data in Figure 9, Figure 11, and Figure 14 are collected from simulations. We also provide the simulation code in AE_figures.ipynb. 
You can play with them and the figures will be automatically generated.
