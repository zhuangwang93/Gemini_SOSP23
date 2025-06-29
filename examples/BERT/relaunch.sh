#!/bin/bash

HOSTFILE=hostfile
TRAIN_SCRIPT=pretrain_bert.py
TRAIN_CONFIG="
5B_template.json
"

JOB_NAME=bert
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
--max_steps 20 \
--print_steps 1 \
--output . \
--comm_profile_steps 12 \
--jump_profile_lines 8 \
--enable_comm_profile \
--snapshot_mode interleave \
--network_bandwidth 80 \
--snapshot_buffer_size 32 \
--span_threshold 100 \
--span_alpha 0.8 \
--max_blocks_in_span 16 \
--load_training_checkpoint saved_models/${JOB_NAME} \
--load_checkpoint_id snapshot \
--save_to_disk \
# --pre_checkpoint \
"

echo $ds_cmd
eval $ds_cmd