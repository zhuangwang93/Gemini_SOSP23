#!/bin/bash

HOSTFILE=hostfile
TRAIN_SCRIPT=pretrain_roberta.py
TRAIN_CONFIG="
roberta_template.json
"

DATETIME=$(date +"%Y-%m-%d_%T")

JOB_NAME=Roberta
OUTPUT_DIR="./${JOB_NAME}"

common_args="\
 \
"

deepspeed="/home/ec2-user/.local/bin/deepspeed"

# running cmd
ds_cmd="\
${deepspeed} --hostfile=${HOSTFILE} ${TRAIN_SCRIPT} \
--deepspeed \
--deepspeed_config $TRAIN_CONFIG \
--job_name ${JOB_NAME} \
--max_steps 32 \
--print_steps 1 \
--output . \
--snapshot_mode interleave \
--comm_profile_steps 24 \
--enable_comm_profile \
--network_bandwidth 150 \
--snapshot_buffer_size 32 \
--span_threshold 10 \
--span_alpha 0.6 \
--max_blocks_in_span 8 \
# --pre_checkpoint \
# --save_to_disk \
"

echo $ds_cmd
eval $ds_cmd