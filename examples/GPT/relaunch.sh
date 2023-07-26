#!/bin/bash

HOSTFILE=hostfile
TRAIN_SCRIPT=pretrain_gpt.py
TRAIN_CONFIG="
gpt_template.json
"

DATETIME=$(date +"%Y-%m-%d_%T")

JOB_NAME=GPT2
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
--snapshot_buffer_size 16 \
--span_threshold 10 \
--span_alpha 0.6 \
--max_blocks_in_span 4 \
--pre_checkpoint \
--load_training_checkpoint saved_models/${JOB_NAME} \
--load_checkpoint_id snapshot \
--save_to_disk \
"

echo $ds_cmd
eval $ds_cmd