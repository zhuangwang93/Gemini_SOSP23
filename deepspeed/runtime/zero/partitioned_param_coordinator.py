"""
"Copyright 2020 The Microsoft DeepSpeed Team.
Licensed under the MIT license.
"""

from dataclasses import dataclass
import collections
from collections import UserDict
from typing import Deque, Set
from torch.cuda import Event, Stream

from deepspeed import comm as dist
from deepspeed.utils.logging import logger
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.utils.debug import debug_module2name_id, debug_param2name_id
from deepspeed.runtime.snapshot.snapshot_global_variables import comm_profiler, snapshot_settings


def debug_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(message)


@instrument_w_nvtx
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse),
                           sub_module.ds_external_parameters())


def iter_params(module: Module, recurse=False) -> Iterable[Parameter]:
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))


class ZeRoTraceMode(Enum):
    # Record trace of the network during a single forward+backward (for training) or forward (for inference)
    RECORD = 1
    # Use recorded network trace to optimize current forward+backward or forward
    COMPLETE = 2
    # Recorded trace does not match current forward+backward or forward pass.
    INVALID = 3


class PartitionedParameterCoordinator:
    """Handles partitioning and gathering of parameters."""
    class __InflightParamRegistry(UserDict):
        """registry for parameters in flight"""
        def __setitem__(self,
                        param: Parameter,
                        handle: AllGatherCoalescedHandle) -> None:
            if param in self.data:
                raise RuntimeError(f"{param.ds_summary()} already in registry")
            if param.ds_status != ZeroParamStatus.INFLIGHT:
                raise RuntimeError(
                    f"attempted to add non-inflight parameter to registry {param.ds_summary()}"
                )
            self.data[param] = handle

    @dataclass
    class __ParamInTrace:
        param: Parameter
        step_id_last_used_at: int

    def __init__(
        self,
        prefetch_bucket_sz: int,
        max_reuse_distance_in_numel: int,
        max_available_parameters_in_numel: int,
        allgather_stream: Stream,
        prefetch_nvme: bool = False,
    ) -> None:
        # mapping of param -> handle for each param that is currently in flight
        self.__inflight_param_registry = __class__.__InflightParamRegistry()
        # keeps track of the number of submodules invoked so far.
        self.__step_id: int = 0
        # network tracing mode
        self.__trace_mode: ZeRoTraceMode = ZeRoTraceMode.RECORD
        # sequence of submodules/parameters in forward pass + backward pass
        self.__submodule_order: Iterable[Module] = []
        self.__param_order: Iterable[__class__.__ParamInTrace] = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
            lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(
            lambda: collections.deque())
        # number of available params, and max number of available params
        self.__n_available_params: int = 0
        self.__max_n_available_params: int = max_available_parameters_in_numel
        # max distance between two use of the module beyond which module is released
        self.__max_reuse_dist_in_numel: int = max_reuse_distance_in_numel
        # queue for parameters to fetch. parameters will be popped off the left
        # side of the dequeue as they are fetched
        self.__param_queue: Deque[__class__.__ParamInTrace] = None
        self.__prefetch_bucket_sz: int = prefetch_bucket_sz
        self.__prefetch_nvme: bool = prefetch_nvme
        self.hierarchy: int = 0

        # stream that will be used for allgather operations
        self.__allgather_stream: Stream = allgather_stream

        # limit the number of fetch events that can be queued at once
        # otherwise, what happens is memory is allocated by the host thread at the
        # time of the call, but not used until later by the asynchronous cuda stream.
        # allowing an infinite number of these to queue up causes a lot of memory
        # pressure that then becomes detrimental to performance.
        # this is a much less elegant way of fixing this vs something like using
        # cudaMallocAsync/cudaFreeAsync. Choosing to not expose this to the user now
        # because ideally in the future its replaced by an async allocation
        # mechanism which doesn't require any configuration by the user.
        self.__ongoing_fetch_events: Deque[Event] = collections.deque()
        # TODO. make this configurable via JSON
        self.__max_ongoing_fetch_events: int = 2

    """Tracing and Tracking
    TODO. consider performing trace before initializing PartitionedParameterCoordinator
    and passing trace results into constructor. This way all the code in here can
    just assume that the trace is complete and the results can be entirely
    immutable.

    Bookkeeping operations used to track where we are in the forward/backward pass
    """

    def _clear_trace_structures(self) -> None:
        self.__submodule_order = []
        self.__param_order = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
            lambda: int(-1e10))
        self.__param_queue = None

    def is_complete_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.COMPLETE

    def is_invalid_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.INVALID

    def is_record_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.RECORD

    def _invalidate_trace(self) -> None:
        if self.is_invalid_trace():
            raise RuntimeError("attempted to invalidate already invalid trace")
        self.__trace_mode = ZeRoTraceMode.INVALID
        self._clear_trace_structures()

    def trace_prologue(self, sub_module: Module) -> None:
        if self.is_complete_trace():
            # sub_module must match expectation else invalidate trace cache
            if sub_module != self.__submodule_order[self.__step_id]:
                expected_module_id = self.__submodule_order[self.__step_id].id
                debug_rank0(
                    f"Invalidate trace cache @ step {self.__step_id}: "
                    f"expected module {expected_module_id}, but got module {sub_module.id}"
                )
                self._invalidate_trace()

    def record_module(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(
                f"attempted to record trace when status = {self.__trace_mode}")

        self.__submodule_order.append(sub_module)
        self.__step_id_module_fetched_for[sub_module.id].append(self.__step_id)

    def record_parameters(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(
                f"attempted to record trace when status = {self.__trace_mode}")

        step_id = self.__step_id_module_fetched_for[sub_module.id].popleft()
        for param in sorted(set(iter_params(sub_module)), key=lambda p: p.ds_id):
            self.__param_order.append(
                __class__.__ParamInTrace(param=param,
                                         step_id_last_used_at=step_id))

    def construct_parameter_trace_from_module_trace(self):
        """use module trace to construct parameter trace"""
        self.__param_order = []
        for sub_module in self.__submodule_order:
            self.record_parameters(sub_module)

    def reset_step(self) -> None:
        """indicate that we have completed one fwd+bwd for the model"""
        if self.__inflight_param_registry:
            raise RuntimeError(
                f"still have inflight params "
                f"{[p.ds_summary for p in self.__inflight_param_registry.keys()]}")

        if not self.is_complete_trace():  # not self.trace_complete:
            # Make sure that recorded parameter and submodule orders are
            # identical across ranks
            assert_ints_same_as_other_ranks([m.id for m in self.__submodule_order])
            assert_ints_same_as_other_ranks([p.param.ds_id for p in self.__param_order])
            assert_ints_same_as_other_ranks(
                [p.step_id_last_used_at for p in self.__param_order])

            if self.is_record_trace():
                # Successfully recorded a trace
                self.construct_parameter_trace_from_module_trace()
                self.__submodule_order = tuple(self.__submodule_order)  # freeze
                self.__param_order = tuple(self.__param_order)  # freeze
                self.__trace_mode = ZeRoTraceMode.COMPLETE
                print_rank_0(
                    f"completed record trace: {[m.id for m in self.__submodule_order]}",
                    force=False)
            else:
                # Enable trace recording for next forward/backward pass
                self.__trace_mode = ZeRoTraceMode.RECORD

        self.__param_queue = collections.deque(self.__param_order)  # reset fetch queue
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(
            lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(
            lambda: collections.deque())
        self.__step_id = 0
        self.__n_available_params = 0

    def _dump_params(self, tag, sub_module, params, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        param_names = [debug_param2name_id(p) for p in params]
        print(
            f'{tag} step = {step_id} mod = {debug_module2name_id(sub_module)} p_names = {param_names}'
        )

    def _dump_param_ids(self, tag, mod_id, p_ids, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        print(f'{tag} mod = {mod_id}, step = {step_id}, p_ids = {p_ids}')

    """Fetch and Release
    Fetching, prefetching, and releasing parameters
    """

    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module) -> None:
        """This method does the following (in order):
        1. kick off fetch for parameters in immediately required sub module
        2. kick off fetch for next few parameters we will need later (prefetch)
        3. block on parameters in immediately required sub module
        """
        debug_rank0(
            f"{self.__step_id}: M{current_submodule.id}({type(current_submodule).__name__}) P{[p.ds_id for p in iter_params(current_submodule)]} "
            + str({
                "avail": f"{self.__n_available_params:.1e}",
                "queue_sz": f"{len(self.__param_queue or [])}",
                "inflight": [p.ds_id for p in self.__inflight_param_registry],
            }))

        params_to_fetch = frozenset(iter_params(current_submodule))

        # kick off all gather for params in the immediately required submodule
        for param in params_to_fetch:
            debug_rank0(f"-fetch: {param.ds_summary()}")
        self.__all_gather_params(params_to_fetch)

        # wait for parameters in the immediately needed submodule to become available
        for param in params_to_fetch:
            param.ds_active_sub_modules.add(current_submodule.id)
            debug_rank0(f"-wait: {param.ds_summary()}")
            if param in self.__inflight_param_registry:
                with torch.cuda.stream(self.__allgather_stream):
                    while self.__ongoing_fetch_events and self.__ongoing_fetch_events[
                            0].query():
                        self.__ongoing_fetch_events.popleft()
                    if len(self.__ongoing_fetch_events
                           ) > self.__max_ongoing_fetch_events:
                        self.__ongoing_fetch_events.popleft().synchronize()

                    self.__inflight_param_registry.pop(param).wait()

                    event = Event()
                    event.record()
                    self.__ongoing_fetch_events.append(event)

            assert param.ds_status == ZeroParamStatus.AVAILABLE, param.ds_summary()
        torch.cuda.current_stream().wait_stream(self.__allgather_stream)
        if not snapshot_settings.is_snapshot_mode():
            torch.cuda.synchronize()

        # kick off parameter prefetches for upcoming modules
        # don't prefetch if we dont have a completed model trace
        if self.is_complete_trace():
            # go through the parameters we need for the current module and pop them
            # off the fetch queue so that they aren't prefetched later.
            # if params have already been popped off the fetch queue by earlier
            # prefetches we won't look for them here
            discarded_from_prefetch_queue = set()
            params_not_already_fetched = set(
                filter(
                    lambda p: self.__most_recent_step_id_param_fetched_for[p] < self.
                    __step_id,
                    params_to_fetch))
            while self.__param_queue and len(discarded_from_prefetch_queue) < len(
                    params_not_already_fetched):
                param_in_trace = self.__param_queue.popleft()
                self.__most_recent_step_id_param_fetched_for[
                    param_in_trace.param] = param_in_trace.step_id_last_used_at
                discarded_from_prefetch_queue.add(param_in_trace.param)

            if discarded_from_prefetch_queue != params_not_already_fetched:
                raise RuntimeError(
                    f"tracing error at step {self.__step_id}: \n"
                    f"module id: {current_submodule.id}, training: {current_submodule.training}\n"
                    f"expected the next {len(params_not_already_fetched)} parameters in the "
                    f"parameter fetch queue to be {tuple(p.ds_summary(use_debug_name=True) for p in params_not_already_fetched)} \n"
                    f"but got \n {tuple(p.ds_summary(use_debug_name=True) for p in discarded_from_prefetch_queue)}."
                )

            def _is_currently_on_nvme(param):
                if param.nvme_swapper is None:
                    return False

                return param.ds_tensor.final_location == OffloadDeviceEnum.nvme \
                    and param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE

            # kick off all gather for params in the next few submodules (prefetch)
            if self.__prefetch_bucket_sz > 0:
                max_params_to_prefetch = min(
                    self.__max_n_available_params - self.__n_available_params,
                    self.__prefetch_bucket_sz)
                params_to_prefetch = set()
                numel_prefetching = 0
                while self.__param_queue and numel_prefetching < max_params_to_prefetch:
                    param_in_trace: __class__.__ParamInTrace = self.__param_queue.popleft(
                    )

                    if _is_currently_on_nvme(param_in_trace.param):
                        # nvme prefetch is handled elsewhere. Need to break here to preserve fetch order
                        self.__param_queue.appendleft(param_in_trace)
                        break

                    do_prefetch = param_in_trace.param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                    if param_in_trace.param in params_to_prefetch:
                        # Avoid duplicates
                        do_prefetch = False

                    self.__most_recent_step_id_param_fetched_for[param_in_trace.param] = \
                        max(self.__most_recent_step_id_param_fetched_for[param_in_trace.param],
                            param_in_trace.step_id_last_used_at)

                    if do_prefetch:
                        params_to_prefetch.add(param_in_trace.param)
                        numel_prefetching += param_in_trace.param.ds_numel

                for param in params_to_prefetch:
                    debug_rank0(f"-prefetch: {param.ds_summary()}")
                self.__all_gather_params(params_to_prefetch)

                if self.__prefetch_nvme:
                    self.__prefetch_nvme_param_partitions()

        self.__step_id += 1

    @instrument_w_nvtx
    @torch.no_grad()
    def release_sub_module(self, submodule: Module) -> None:
        """release the parameters of a sub module, assuming they meet conditions to
        be released."""
        params_to_release = (self.__params_to_release(submodule,
                                                      self.__step_id)
                             if self.is_complete_trace() else set(
                                 p.ds_id for p in iter_params(submodule)))
        for param in iter_params(submodule):
            param.ds_active_sub_modules.discard(submodule.id)
            if param.ds_id in params_to_release and not param.is_external_param:
                self.__release_param(param)

    @instrument_w_nvtx
    @torch.no_grad()
    def release_and_reset_all(self, module: Module) -> None:
        """release all module parameters"""
        for param in iter_params(module, recurse=True):
            if param in self.__inflight_param_registry:
                raise RuntimeError(f"param {param.ds_summary()} still in flight")

            # TODO. make this throw if if there are still active submodules. currently
            # there's a hook execution issue
            param.ds_active_sub_modules.clear()
            self.__release_param(param)

        for param in iter_params(module, recurse=True):
            if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                raise RuntimeError(f"{param.ds_summary()} expected to be released")

    @instrument_w_nvtx
    def __all_gather_params(self, params: Set[Parameter]) -> None:
        """for each partitioned parameter, kick off an async allgather and store
        the work handle for the in flight parameters."""
        partitioned_params = []
        for param in params:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                partitioned_params.append(param)
                self.__n_available_params += param.ds_numel

        if partitioned_params:            
            # with torch.cuda.stream(self.__allgather_stream):
            #     handle = partitioned_params[0].all_gather_coalesced(partitioned_params)

            with torch.cuda.stream(self.__allgather_stream):
                start = torch.cuda.Event(enable_timing=True)
                start.record(stream=self.__allgather_stream)
                handle = partitioned_params[0].all_gather_coalesced(partitioned_params)
                end = torch.cuda.Event(enable_timing=True)
                end.record(stream=self.__allgather_stream)
                # use stream.synchronize() for accurate profiling. 
                # We observe that stream.synchronize() can improve the iteration time
                self.__allgather_stream.synchronize()
                comm_profiler.add_start_allgather_event(start)
                comm_profiler.add_end_allgather_event(end)
                
            if snapshot_settings.is_snapshot_mode():
                from deepspeed.runtime.snapshot.snapshot_comm import cpu_snapshot
                cpu_snapshot.set_allgather_stream(self.__allgather_stream)
                cpu_snapshot.remote_snapshot(handle.get_handle(), self.__allgather_stream)

            for param in partitioned_params:
                assert param.ds_status == ZeroParamStatus.INFLIGHT, param.ds_summary()
                self.__inflight_param_registry[param] = handle

            # Release swap buffers for persisted params on nvme since they will never be partitioned or evicted from GPU
            swap_persisted_params = [
                p for p in partitioned_params
                if p.ds_persist and p.ds_tensor.final_location == OffloadDeviceEnum.nvme
            ]
            if swap_persisted_params:
                swap_persisted_params[
                    0].nvme_swapper.remove_partition_and_release_buffers(
                        swap_persisted_params)
                
            if not snapshot_settings.is_snapshot_mode():
                torch.cuda.synchronize()
            

    @instrument_w_nvtx
    def __release_param(self, param: Parameter) -> None:
        if param.ds_status == ZeroParamStatus.AVAILABLE and not param.ds_active_sub_modules:
            debug_rank0(f"-release: {param.ds_summary()}")
            param.partition()
            self.__n_available_params -= param.ds_numel

    @instrument_w_nvtx
    @functools.lru_cache(maxsize=None)
    def __params_to_release(self,
                            submodule_to_release: Module,
                            step_id: int) -> Set[int]:
        if not self.is_complete_trace():
            raise RuntimeError("expected trace to be complete")

        params_to_release = set(p.ds_id for p in iter_params(submodule_to_release)
                                if not p.ds_persist)

        # Problem: When prefetcher scans the param trace, it skips AVAILABLE params.
        # This creates issues if those params are released before the skipped uses:
        # 1) It hurts performance as the skipped uses are never prefetched.
        # 2) For nvme params, we run out of swap buffers because the prefetch order
        # diverges from the trace.
        # Solution: Don't release params whose reuse was skipped by prefetch. This is
        # possible because we detect such skips during prefetch and mark those params.
        for param in iter_params(submodule_to_release):
            if self.__most_recent_step_id_param_fetched_for[param] > step_id:
                params_to_release.discard(param.ds_id)

        # examine all modules within `max_reuse_dist_in_numel` of the current step,
        # if we see any of the candidate parameters to be released reoccur while
        # doing this, remove them from the set of parameters to release.
        params_traversed = 0
        for module in self.__submodule_order[step_id:]:
            if params_traversed >= self.__max_reuse_dist_in_numel:
                break
            for param in iter_params(module):
                params_to_release.discard(param.ds_id)
                params_traversed += param.ds_numel

        return params_to_release

    @instrument_w_nvtx
    def __prefetch_nvme_param_partitions(self) -> None:
        """swap in parameter partitions from nvme for those parameters that will be used
        after the ones that are already being prefetched into full parameters
        """
        if not self.is_complete_trace():
            return

        numel_in_flight = sum(param.ds_numel for param in self.__inflight_param_registry)

        numel_considered = 0
        swap_in_params = []
        for param_in_trace in self.__param_queue:
            param = param_in_trace.param
            if param.nvme_swapper is None:
                continue
            if (numel_considered > 2 * numel_in_flight or len(swap_in_params) >=
                    param.nvme_swapper.available_swap_in_buffers()):
                break
            if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                swap_in_params.append(param)
            numel_considered += param.ds_numel

        if swap_in_params:
            swap_in_params[0].nvme_swapper.swap_in(swap_in_params, async_op=True)
