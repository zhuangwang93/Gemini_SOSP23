#!/bin/bash

HOSTFILE="../hostfile"
TRAIN_SCRIPT=deepspeed_train.py
TRAIN_CONFIG="bert_large.json"

deepspeed="/home/ec2-user/.local/bin/deepspeed"

${deepspeed} --hostfile=${HOSTFILE} \
    ${TRAIN_SCRIPT} \
    --deepspeed \
    --deepspeed_config  deepspeed_adam_config.json \
    --cf ${TRAIN_CONFIG} \
    --max_seq_length 512 \
    --max_steps 24 \
    --print_steps 1 \
    --use_nvidia_dataset \
    --deepspeed_transformer_kernel \
    --output . \
    --comm_profile_steps 16 \
    --jump_profile_lines 12 \
    --enable_comm_profile \
    --snapshot_mode interleave \
    --network_bandwidth 80 \
    --snapshot_buffer_size 32 \
    --span_threshold 100 \
    --span_alpha 0.8 \
    --max_blocks_in_span 16 \
    # --save_to_disk \
    # --enable_snapshot_profile \