export NCCL_SOCKET_IFNAME=eth0

deepspeed="/home/ec2-user/.local/bin/deepspeed"

${deepspeed} --hostfile=hostfile  \
    deepspeed_train.py \
    --deepspeed \
    --deepspeed_config  deepspeed_adam_config.json \
    --cf bert_base.json \
    --max_seq_length 512 \
    --max_steps 64 \
    --print_steps 1 \
    --use_nvidia_dataset \
    --deepspeed_transformer_kernel \
    --output . \
    --load_training_checkpoint saved_models/bing_bert_base_seq \
    --load_checkpoint_id snapshot \
    --snapshot_mode naive \
    # --enable_comm_profile \
    # --comm_profile_steps 20 \
    # --enable_snapshot_profile \

# ${deepspeed} --num_nodes 1 \
#     deepspeed_train.py \
#     --deepspeed \
#     --deepspeed_config  deepspeed_adam_config.json \
#     --cf bert_base.json \
#     --max_seq_length 128 \
#     --max_steps 64 \
#     --print_steps 1 \
#     --use_nvidia_dataset \
#     --deepspeed_transformer_kernel \
#     --output . \
#     --load_training_checkpoint saved_models/bing_bert_base_seq \
#     --load_checkpoint_id snapshot \
#     --enable_snapshot \
#     --enable_snapshot_profile \