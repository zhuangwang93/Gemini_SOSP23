export FI_PROVIDER="efa"
export NCCL_SOCKET_IFNAME=eth
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_ALGO=Ring

# nsys profile -t nvtx,cuda 

mpirun -np 16 -N 8 -hostfile hostfile \
    --mca plm_rsh_no_tree_spawn 1 \
    -mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 \
    --mca pml ^cm \
    -bind-to none \
    python3 deepspeed_train.py \
    --deepspeed \
    --deepspeed_mpi \
    --deepspeed_config  deepspeed_adam_config.json \
    --cf bert_large.json \
    --max_seq_length 512 \
    --max_steps 48 \
    --print_steps 1 \
    --use_nvidia_dataset \
    --deepspeed_transformer_kernel \
    --output . \
    --snapshot_mode naive \
    --comm_profile_steps 32 \
    --enable_comm_profile \
    --enable_snapshot_profile \
