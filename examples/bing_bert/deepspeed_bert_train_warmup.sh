export NCCL_SOCKET_IFNAME=eth0

deepspeed="/home/ec2-user/.local/bin/deepspeed"

${deepspeed} --num_nodes 1 \
    deepspeed_train.py \
    --deepspeed \
    --deepspeed_config  deepspeed_adam_config.json \
    --cf bert_base.json \
    --max_seq_length 128 \
    --max_steps 16 \
    --print_steps 1 \
    --use_nvidia_dataset \
    --deepspeed_transformer_kernel \
    --output . \