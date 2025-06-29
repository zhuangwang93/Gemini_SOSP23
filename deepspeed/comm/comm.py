"""
    Copyright 2021 The Microsoft DeepSpeed Team

    DeepSpeed Communication Package: deepspeed.comm

    deepspeed.comm
        -- import and use deepspeeed.ops.comm
        -- use torch.distributed directly if both this package and torch.distributed use the same NCCL version
        -- use custom collectives
            -- can either use torch.dist or ds.ops.comm?

        Note: the old 1-bit compressed allreduce variants that resided in deepspeed.runtime.comm will be moved here as well.

    deepspeed.comm API
        -- must be kept fully compatible (same signatures) as torch.dist API to ensure backward/cross-framework compatibility.
        -- e.g. if a client code used
            from deepspeed import comm as dist

            instead of
            import torch.distributed as dist

            The code should work without breaking any of the public torch.distributed functionality

    Future:
        -- deepspeed groups API should be brought into ds.comm
"""

from enum import Enum
import torch
import os

from ..constants import TORCH_DISTRIBUTED_DEFAULT_PORT, default_pg_timeout
from .constants import *


class ReduceOp(Enum):
    SUM = 0
    PRODUCT = 1
    MIN = 2
    MAX = 3
    BAND = 4
    BOR = 5
    BXOR = 6
    AVG = 7
    UNUSED = 8


from deepspeed.utils.comms_logging import CommsLogger
from deepspeed.utils import timer, get_caller_func
from deepspeed.comm.torch import TorchBackend

from deepspeed import utils
from datetime import timedelta

# Current deepspeed.comm backend (cdb) global object for simple access by client code
use_ds_backend = False
cdb = None

# Create global timer for ops
timers = timer.SynchronizedWallClockTimer()
timer_summary = {}

comms_logger = CommsLogger()

# Ensure we don't warn about base collectives more than once
has_warned_all_gather = False
has_warned_reduce_scatter = False

# Maintain objects of all initialized ds backends and assign them using the API functions in this file
nccl_backend = None
mpi_backend = None

# This should be set here so all rank/size information from the launcher can be propagated
from deepspeed.comm.utils import *


def _configure_using_config_file(config):
    if config.comms_logger_enabled:
        comms_logger.configure(config)


def configure(
    deepspeed_config=None,
    enabled=None,
    prof_all=None,
    prof_ops=None,
    verbose=None,
    debug=None,
):

    if deepspeed_config is not None:
        _configure_using_config_file(deepspeed_config.comms_config)

    if enabled is not None:
        comms_logger.enabled = enabled

    if prof_all is not None:
        comms_logger.prof_all = prof_all

    if prof_ops is not None:
        comms_logger.prof_ops = prof_ops

    if verbose is not None:
        comms_logger.verbose = verbose

    if debug is not None:
        comms_logger.debug = debug


# Logging wrapper for timing ops
def timed_op(func):
    def log_wrapper(*args, **kwargs):
        # Add enabled flag so that overhead to each comm op is two if conditions at most
        if comms_logger.enabled:
            if ('prof' in kwargs and kwargs['prof']) or comms_logger.prof_all or (
                    'log_name' in kwargs
                    and kwargs['log_name'] in comms_logger.prof_ops):
                # Need func args for their defaults
                func_args = get_default_args(func)
                func_args.update(kwargs)
                msg_size = get_msg_size_from_args(func, *args, **kwargs)
                log_name = get_debug_log_name(func_args, comms_logger.debug)
                timers(log_name).start()
        # Return the op, then stop the op's timer
        try:
            return func(*args, **kwargs)
        finally:
            if comms_logger.enabled:
                # Need to make op blocking for accurate logging
                torch.cuda.synchronize()
                # If we're using MPI, we can't simply sync the stream
                if cdb.using_mpi:
                    cdb.barrier()
                if ('prof' in kwargs and kwargs['prof']) or comms_logger.prof_all or (
                        'log_name' in kwargs
                        and kwargs['log_name'] in comms_logger.prof_ops):
                    log_name = get_debug_log_name(func_args, comms_logger.debug)
                    raw_name = func.__name__
                    timers(log_name).stop()
                    # need temp var since 'elapsed' resets events
                    time_elapsed = timers(log_name).elapsed(reset=False)
                    comms_logger.append(raw_name, log_name, time_elapsed, msg_size)

    return log_wrapper


# For compatibility with torch distributed's init_process_group, we shall retain the signature from PyTorch code.
# DeepSpeed NCCL/MPI backend may not need all these params as we will have our own implementation.
# Please read full torch.distributed API docs from https://pytorch.org/docs/stable/distributed.html


# UNUSED: Future helper function to initialize DS backends
def init_deepspeed_backend(ds_backend):
    global cdb
    global nccl_backend
    global mpi_backend
    global use_ds_backend

    if ds_backend == NCCL_BACKEND:
        utils.logger.warn("NCCL backend in DeepSpeed not yet implemented")
    elif ds_backend == MPI_BACKEND:
        utils.logger.warn("MPI backend in DeepSpeed not yet implemented")
    elif ds_backend == GLOO_BACKEND:
        utils.logger.warn("Gloo backend in DeepSpeed not yet implemented")
    else:
        utils.logger.warn(f"DeepSpeed does not support {ds_backend} backend")


def is_initialized():
    #assert cdb is not None, 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb is None:
        return False
    else:
        return cdb.is_initialized()


def destroy_process_group(group=None):
    global cdb
    return cdb.destroy_process_group(group=group)


def new_group(ranks):
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.new_group(ranks)


def is_available() -> bool:

    # Returns ``True`` if the deepspeed comm package is available.

    # TODO: load other ops. Clients including deepspeed itself should use deepspeed.comm to import
    # any communication related primitives from this package.
    # use hasattr(deepspeed.csrc.ops, "_comm") or something
    return True


def set_backend(backend):
    if not use_ds_backend:
        utils.logger.error(
            "DeepSpeed communication backend is required. Please use deepspeed.comm.init_distributed(backend, use_deepspeed=True) to use this functionality"
        )
        raise RuntimeError(
            'Error: Custom DeepSpeed backend called without initializing DeepSpeed distributed.'
        )

    global cdb
    global nccl_backend
    global mpi_backend

    try:
        if backend_name == NCCL_BACKEND:
            if nccl_backend is not None and nccl_backend.is_initialized():
                cdb = nccl_backend
        elif backend_name == MPI_BACKEND:
            if mpi_backend is not None and mpi_backend.is_initialized():
                cdb = mpi_backend
    except Exception as inst:
        print(inst)


@timed_op
def broadcast(tensor,
              src,
              group=None,
              async_op=False,
              prof=False,
              log_name='broadcast',
              debug=get_caller_func()):
    global cdb
    return cdb.broadcast(tensor=tensor, src=src, group=group, async_op=async_op)


@timed_op
def all_gather(tensor_list,
               tensor,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_gather',
               debug=get_caller_func()):
    global cdb
    return cdb.all_gather(tensor_list=tensor_list,
                          tensor=tensor,
                          group=group,
                          async_op=async_op)


def has_reduce_scatter_base():
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    assert cdb.has_reduce_scatter_base is not None, 'has_reduce_scatter_base is not yet defined'
    return cdb.has_reduce_scatter_base


def reduce_scatter_fn(output_tensor,
                      tensor,
                      group=None,
                      async_op=False,
                      prof=False,
                      debug=get_caller_func()):
    global cdb
    global has_warned_reduce_scatter
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb.has_reduce_scatter_base:
        return reduce_scatter_base(output_tensor,
                                   tensor,
                                   group=group,
                                   async_op=async_op,
                                   prof=prof,
                                   debug=debug)
    else:
        if not has_warned_reduce_scatter:
            utils.logger.warning(
                "unable to find torch.distributed._reduce_scatter_base. will fall back to "
                "torch.distributed.all_gather which will result in suboptimal performance. "
                "please consider upgrading your pytorch installation.")
            has_warned_reduce_scatter = True
        input_tensor_lst = list(torch.chunk(tensor, cdb.get_world_size(group)))
        return reduce_scatter(output_tensor,
                              input_tensor_lst,
                              group=group,
                              async_op=async_op,
                              prof=prof,
                              debug=debug)


@timed_op
def reduce_scatter_base(output_tensor,
                        tensor,
                        group=None,
                        async_op=False,
                        prof=False,
                        log_name='reduce_scatter_base',
                        debug=get_caller_func()):
    global cdb
    return cdb.reduce_scatter_base(output_tensor=output_tensor,
                                   input_tensor=tensor,
                                   group=group,
                                   async_op=async_op)


@timed_op
def all_gather_base(output_tensor,
                    tensor,
                    group=None,
                    async_op=False,
                    prof=False,
                    log_name='all_gather_base',
                    debug=get_caller_func()):
    global cdb
    return cdb.all_gather_base(output_tensor=output_tensor,
                               input_tensor=tensor,
                               group=group,
                               async_op=async_op)


def has_allgather_base():
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    assert cdb.has_allgather_base is not None, 'has_allgather_base is not yet defined'
    return cdb.has_allgather_base


def allgather_fn(output_tensor,
                 input_tensor,
                 group=None,
                 async_op=False,
                 debug=get_caller_func()):
    global cdb
    global has_warned_all_gather
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    if cdb.has_allgather_base:
        return all_gather_base(output_tensor,
                               input_tensor,
                               group=group,
                               async_op=async_op,
                               debug=debug)
    else:
        if not has_warned_all_gather and get_rank() == 0:
            utils.logger.warning(
                "unable to find torch.distributed._all_gather_base. will fall back to "
                "torch.distributed.all_gather which will result in suboptimal performance. "
                "please consider upgrading your pytorch installation.")
            has_warned_all_gather = True
        output_tensors = list(torch.chunk(output_tensor, cdb.get_world_size(group)))
        return all_gather(output_tensors,
                          input_tensor,
                          group=group,
                          async_op=async_op,
                          debug=debug)


@timed_op
def all_to_all_single(output,
                      tensor,
                      output_split_sizes=None,
                      input_split_sizes=None,
                      group=None,
                      async_op=False,
                      prof=False,
                      log_name='all_to_all_single',
                      debug=get_caller_func()):
    global cdb
    return cdb.all_to_all_single(output=output,
                                 input=tensor,
                                 output_split_sizes=output_split_sizes,
                                 input_split_sizes=input_split_sizes,
                                 group=group,
                                 async_op=async_op)


@timed_op
def send(tensor,
         dst,
         group=None,
         tag=0,
         prof=False,
         log_name='send',
         debug=get_caller_func()):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


@timed_op
def recv(tensor,
         src=None,
         group=None,
         tag=0,
         prof=False,
         log_name='recv',
         debug=get_caller_func()):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


@timed_op
def isend(tensor,
          dst,
          group=None,
          tag=0,
          prof=False,
          log_name='isend',
          debug=get_caller_func()):
    global cdb
    return cdb.send(tensor=tensor, dst=dst, group=group, tag=tag)


@timed_op
def irecv(tensor,
          src=None,
          group=None,
          tag=0,
          prof=False,
          log_name='irecv',
          debug=get_caller_func()):
    global cdb
    return cdb.recv(tensor=tensor, src=src, group=group, tag=tag)


@timed_op
def gather(tensor,
           gather_list=None,
           dst=0,
           group=None,
           async_op=False,
           prof=False,
           log_name='gather',
           debug=get_caller_func()):
    global cdb
    return cdb.gather(tensor=tensor,
                      gather_list=gather_list,
                      dst=dst,
                      group=group,
                      async_op=async_op)


@timed_op
def scatter(tensor,
            scatter_list=None,
            src=0,
            group=None,
            async_op=False,
            prof=False,
            log_name='scatter',
            debug=get_caller_func()):
    global cdb
    return cdb.scatter(tensor=tensor,
                       scatter_list=scatter_list,
                       src=src,
                       group=group,
                       async_op=async_op)


@timed_op
def barrier(group=None, prof=False, log_name='barrier', debug=get_caller_func()):
    global cdb
    return cdb.barrier()


def log_summary():
    global cdb
    barrier(log_name='log_summary_barrier')
    if cdb.get_rank() == 0:
        comms_logger.log_all()
    barrier(log_name='log_summary_barrier')


@timed_op
def reduce(tensor,
           dst,
           op=ReduceOp.SUM,
           group=None,
           async_op=False,
           prof=False,
           log_name='reduce',
           debug=get_caller_func()):
    global cdb
    return cdb.reduce(tensor=tensor, dst=dst, op=op, group=group, async_op=async_op)


@timed_op
def reduce_scatter(output,
                   input_list,
                   op=ReduceOp.SUM,
                   group=None,
                   async_op=False,
                   prof=False,
                   log_name='reduce_scatter',
                   debug=get_caller_func()):
    global cdb
    return cdb.reduce_scatter(output=output,
                              input_list=input_list,
                              op=op,
                              group=group,
                              async_op=async_op)


@timed_op
def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=None,
               async_op=False,
               prof=False,
               log_name='all_reduce',
               debug=get_caller_func()):
    #if profile_comm:
    # context of the timers?
    # timers.start()
    # TensorBoard logging for comm calls.?
    global cdb
    #print(f'op = {op}, cdb= {cdb.name}')
    return cdb.all_reduce(tensor, op, group, async_op)


def get_world_group():
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_world_group()


def get_world_size(group=None) -> int:
    """
    Returns the number of processes in the current process group
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The world size of the process group
        -1, if not part of the group
    """
    global cdb

    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_world_size(group)


def get_rank(group=None):
    """
    Returns the rank of the current process in the provided ``group`` or the
    default group if none was provided.
    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.
    Args:
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
    Returns:
        The rank of the process group
        -1, if not part of the group
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_rank(group)


def get_local_rank():
    """
        Helper function to get local rank after a backend has been set and initialized
        Args:
            None
        Returns:
            local rank (= GPU device ID)
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return get_local_rank_from_launcher()



def get_local_size():
    """
        Helper function to get local size after a backend has been set and initialized
        Args:
            None
        Returns:
            local size (= the number of GPUs in one machine)
    """
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return get_local_size_from_launcher()


def get_global_rank(group=None, group_rank=0):
    global cdb
    assert cdb is not None and cdb.is_initialized(), 'DeepSpeed backend not set, please initialize it using init_process_group()'
    return cdb.get_global_rank(group, group_rank)


# Main DeepSpeed Comms. public API.
def init_distributed(dist_backend="nccl",
                     auto_mpi_discovery=True,
                     distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT,
                     verbose=True,
                     timeout=default_pg_timeout,
                     init_method=None,
                     dist_init_required=None,
                     config=None):
    ''' Initialize dist backend, potentially performing MPI discovery if needed

    Arguments:
        dist_backend: Optional (str). torch distributed backend, e.g., nccl, mpi, gloo
        auto_mpi_discovery Optional (bool). if distributed environment variables are not set, attempt to discover them from MPI
        distributed_port: Optional (int). torch distributed backend port
        verbose: Optional (bool). verbose logging
        timeout: Optional (timedelta). Timeout for operations executed against the process group. Default value equals 30 minutes.
        init_method: Optional (string). Torch distributed, URL specifying how to initialize the process group. Default is “env://” if no init_method or store is specified.
        config: Optional (dict). DeepSpeed configuration for setting up comms options (e.g. Comms profiling)
    '''
    global cdb

    configure(deepspeed_config=config)

    if dist_init_required is None:
        dist_init_required = cdb is None or not cdb.is_initialized()

    if cdb is None and torch.distributed.is_initialized():
        # The user initialized torch.dist themselves, create cdb and short-circuit
        cdb = TorchBackend(dist_backend, timeout, init_method)
        return

    if dist_init_required is False:
        assert (
            cdb is not None and cdb.is_initialized() is True
        ), "Distributed backend is not initialized. Please set dist_init_required to True or initialize before calling deepspeed.initialize()"
    else:
        # Initialize torch distributed if needed
        required_env = ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
        if auto_mpi_discovery and not all(map(lambda v: v in os.environ, required_env)):
            if verbose:
                utils.logger.info(
                    "Not using the DeepSpeed or dist launchers, attempting to detect MPI environment..."
                )
            if in_aml() and not in_dlts():
                patch_aml_env_for_torch_nccl_backend(verbose=verbose)
            elif in_aws_sm():
                patch_aws_sm_env_for_torch_nccl_backend(verbose=verbose)
            else:
                mpi_discovery(distributed_port=distributed_port, verbose=verbose)

        if cdb is not None and cdb.is_initialized():
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.info('Distributed backend already initialized')
        else:
            assert isinstance(timeout, timedelta)
            if int(os.getenv('RANK', '0')) == 0:
                utils.logger.info(
                    'Initializing TorchBackend in DeepSpeed with backend {}'.format(
                        dist_backend))
            # Create a torch backend object, initialize torch distributed, and assign to cdb
            cdb = TorchBackend(dist_backend, timeout, init_method)


def mpi_discovery(distributed_port=TORCH_DISTRIBUTED_DEFAULT_PORT, verbose=True):
    '''
    Discovery MPI environment via mpi4py and map to relevant dist state
    '''
    from mpi4py import MPI
    import subprocess
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    master_addr = None
    if rank == 0:
        hostname_cmd = ["hostname -I"]
        result = subprocess.check_output(hostname_cmd, shell=True)
        master_addr = result.decode('utf-8').split()[0]
    master_addr = comm.bcast(master_addr, root=0)

    # Determine local rank by assuming hostnames are unique
    proc_name = MPI.Get_processor_name()
    all_procs = comm.allgather(proc_name)
    local_rank = sum([i == proc_name for i in all_procs[:rank]])

    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(distributed_port)

    if verbose:
        utils.logger.info(
            "Discovered MPI settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'],
                    os.environ['LOCAL_RANK'],
                    os.environ['WORLD_SIZE'],
                    os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))

    if cdb is not None and cdb.is_initialized():
        assert cdb.get_rank() == rank, "MPI rank {} does not match torch rank {}".format(
            rank, cdb.get_rank())
        assert cdb.get_world_size() == world_size, "MPI world size {} does not match torch world size {}".format(
            world_size, cdb.get_world_size())


def in_aml():
    # Are we running inside an Azure Machine Learning (AML) environment?
    return 'AZUREML_EXPERIMENT_ID' in os.environ


def in_aws_sm():
    # Are we running inside an AWS SageMaker environment?
    return 'SM_TRAINING_ENV' in os.environ


def in_dlts():
    # Are we running on a DLTS cluster?
    return 'DLTS_JOB_ID' in os.environ


def patch_aml_env_for_torch_nccl_backend(master_port=6105, verbose=True):
    """Helper routine to get and set environment variables.
    This is adapted from Azure ML's documentation available from:
    https://azure.github.io/azureml-web/docs/cheatsheet/distributed-training/#environment-variables-from-openmpi
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
    single_node = int(os.environ["OMPI_COMM_WORLD_LOCAL_SIZE"]) == int(
        os.environ["WORLD_SIZE"])

    if not single_node:
        master_node_params = os.environ["AZ_BATCH_MASTER_NODE"].split(":")
        os.environ["MASTER_ADDR"] = master_node_params[0]
        # Do not overwrite master port with that defined in AZ_BATCH_MASTER_NODE
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(master_port)
    else:
        os.environ["MASTER_ADDR"] = os.environ["AZ_BATCHAI_MPI_MASTER_NODE"]
        os.environ["MASTER_PORT"] = DEFAULT_AML_MASTER_PORT

    if verbose:
        utils.logger.info("NCCL_SOCKET_IFNAME original value = {}".format(
            os.environ["NCCL_SOCKET_IFNAME"]))

    os.environ["NCCL_SOCKET_IFNAME"] = DEFAULT_AML_NCCL_SOCKET_IFNAME
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]

    if verbose:
        utils.logger.info(
            "Discovered AzureML settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'],
                    os.environ['LOCAL_RANK'],
                    os.environ['WORLD_SIZE'],
                    os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))


def patch_aws_sm_env_for_torch_nccl_backend(verbose=True):
    """Helper routine to get and set environment variables when running inside an AWS SageMaker environment.
    """
    os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
    os.environ['LOCAL_RANK'] = os.environ["OMPI_COMM_WORLD_LOCAL_RANK"]
    os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]

    if verbose:
        utils.logger.info(
            "Discovered AWS SageMaker settings of world_rank={}, local_rank={}, world_size={}, master_addr={}, master_port={}"
            .format(os.environ['RANK'],
                    os.environ['LOCAL_RANK'],
                    os.environ['WORLD_SIZE'],
                    os.environ['MASTER_ADDR'],
                    os.environ['MASTER_PORT']))
