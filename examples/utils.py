import sys
import argparse


def get_argument_parser():
    parser = argparse.ArgumentParser()

    # Required_parameter
    parser.add_argument(
        "--config-file",
        "--cf",
        help="pointer to the configuration file of the experiment",
        type=str)
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written."
    )

    # Optional Params
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help=
        "The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded."
    )
    parser.add_argument(
        "--max_predictions_per_seq",
        "--max_pred",
        default=80,
        type=int,
        help=
        "The maximum number of masked tokens in a sequence to be predicted.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument(
        "--do_lower_case",
        default=True,
        action='store_true',
        help=
        "Whether to lower case the input text. True for uncased models, False for cased models."
    )
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument('--use_pretrain',
                        default=False,
                        action='store_true',
                        help="Whether to use Bert Pretrain Weights or not")

    parser.add_argument(
        '--refresh_bucket_size',
        type=int,
        default=1,
        help=
        "This param makes sure that a certain task is repeated for this time steps to \
                            optimise on the back propogation speed with APEX's DistributedDataParallel"
    )
    parser.add_argument('--finetune',
                        default=False,
                        action='store_true',
                        help="Whether to finetune only")

    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='LE',
        help=
        'Choices LE, EE, EP (L: Linear, E: Exponetial, P: Polynomial warmup and decay)'
    )

    parser.add_argument('--lr_offset',
                        type=float,
                        default=0.0,
                        help='Offset added to lr.')

    parser.add_argument(
        '--load_training_checkpoint',
        '--load_cp',
        type=str,
        default=None,
        help=
        "This is the path to the TAR file which contains model+opt state_dict() checkpointed."
    )
    parser.add_argument(
        '--load_checkpoint_id',
        '--load_cp_id',
        type=str,
        default=None,
        help='Checkpoint identifier to load from checkpoint path')
    parser.add_argument(
        '--job_name',
        type=str,
        default=None,
        help="This is the path to store the output and TensorBoard results.")

    parser.add_argument(
        '--rewarmup',
        default=False,
        action='store_true',
        help='Rewarmup learning rate after resuming from a checkpoint')

    parser.add_argument(
        '--max_steps',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size to complete.'
    )

    parser.add_argument(
        '--max_steps_per_epoch',
        type=int,
        default=sys.maxsize,
        help=
        'Maximum number of training steps of effective batch size within an epoch to complete.'
    )

    parser.add_argument('--print_steps',
                        type=int,
                        default=100,
                        help='Interval to print training details.')

    parser.add_argument(
        '--data_path_prefix',
        type=str,
        default="",
        help=
        "Path to prefix data loading, helpful for AML and other environments")

    parser.add_argument(
        '--validation_data_path_prefix',
        type=str,
        default=None,
        help=
        "Path to prefix validation data loading, helpful if pretraining dataset path is different"
    )

    parser.add_argument('--deepspeed_transformer_kernel',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed transformer kernel to accelerate.')

    parser.add_argument(
        '--stochastic_mode',
        default=False,
        action='store_true',
        help='Use stochastic mode for high-performance transformer kernel.')

    parser.add_argument(
        '--ckpt_to_save',
        nargs='+',
        type=int,
        help=
        'Indicates which checkpoints to save, e.g. --ckpt_to_save 160 161, by default all checkpoints are saved.'
    )

    parser.add_argument(
        '--attention_dropout_checkpoint',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to checkpoint dropout output.'
    )
    parser.add_argument(
        '--normalize_invertible',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to perform invertible normalize backpropagation.'
    )
    parser.add_argument(
        '--gelu_checkpoint',
        default=False,
        action='store_true',
        help=
        'Use DeepSpeed transformer kernel memory optimization to checkpoint GELU activation.'
    )
    parser.add_argument('--deepspeed_sparse_attention',
                        default=False,
                        action='store_true',
                        help='Use DeepSpeed sparse self attention.')

    parser.add_argument('--use_nvidia_dataset',
                        default=False,
                        action='store_true',
                        help='Use Nvidia pretraining dataset.')

    parser.add_argument('--progressive_layer_drop',
                        default=False,
                        action='store_true',
                        help="Whether to enable progressive layer dropping or not")

    # snapshot related arguments
    parser.add_argument('--snapshot_path',
                        type=str,
                        default="snapshot",
                        help="snapshot store path")
    parser.add_argument('--snapshot_mode',
                        type=str,
                        default="none",
                        help="set remote snapshot mode")
    parser.add_argument('--enable_snapshot_profile',
                        default=False,
                        action='store_true',
                        help="enable to profile the performance of remote snapshot")
    parser.add_argument('--enable_comm_profile',
                        default=False,
                        action='store_true',
                        help="enable to profile the communication timeline")
    parser.add_argument('--jump_profile_lines',
                        type=int,
                        default=10,
                        help="jump profile lines")
    parser.add_argument('--comm_profile_steps',
                        type=int,
                        default=20,
                        help="steps to profile the communication timeline")
    parser.add_argument('--pre_checkpoint',
                        default=False,
                        action='store_true',
                        help="checkpoint states before training")
    parser.add_argument('--network_bandwidth',
                        type=int,
                        default=100,
                        help="the network bandwidth connection instances")
    parser.add_argument('--snapshot_buffer_size',
                        type=int,
                        default=4,
                        help="the GPU size for snapshot (M FP32)")
    parser.add_argument('--span_threshold',
                        type=int,
                        default=10,
                        help="the threshold to filter small time spans")
    parser.add_argument('--span_alpha',
                        type=float,
                        default=0.6,
                        help="the argument to consider the variance of profiled spans")
    parser.add_argument('--max_blocks_in_span',
                        type=int,
                        default=2,
                        help="the maximum number of blocks in a span")
    parser.add_argument('--save_to_disk',
                        default=False,
                        action='store_true',
                        help="save the snapshot to disk (for test only)")
    return parser


def is_time_to_exit(args, epoch_steps=0, global_steps=0):
    return (epoch_steps >= args.max_steps_per_epoch) or \
            (global_steps >= args.max_steps)



def set_snapshot_settings(args):
    from deepspeed.runtime.snapshot.snapshot_comm import cpu_snapshot
    from deepspeed.runtime.snapshot.snapshot_global_variables import snapshot_settings, training_profiler, snapshot_profiler, comm_profiler
    snapshot_settings.set_snapshot_mode(args.snapshot_mode)
    snapshot_settings.set_profile_mode(args.enable_snapshot_profile)
    snapshot_settings.set_comm_profile_mode(args.enable_comm_profile)
    snapshot_settings.set_comm_profile_steps(args.comm_profile_steps)
    cpu_snapshot.set_checkpoint_setup(args)
    cpu_snapshot.set_snapshot_mode(snapshot_settings.get_snapshot_mode())

    return cpu_snapshot, snapshot_settings, training_profiler, snapshot_profiler, comm_profiler
