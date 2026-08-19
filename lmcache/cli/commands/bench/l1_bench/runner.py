# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner for GPU-to-L1 Store / Load operations.

Each benchmark worker maintains a bounded queue of incomplete requests. Every
request owns a GPU block slot, producer event, and device-aware future. When a
worker reaches ``max_in_flight_requests``, the oldest request is completed
before its slot is reused. A measured operation ends only after every future
has observed both the MP response and the device completion event.
"""

# Future
from __future__ import annotations

# Standard
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from typing import Any
import os
import threading
import time

# Third Party
import torch
import zmq

# First Party
from lmcache import torch_dev
from lmcache.cli.commands.bench.server_bench.helpers import (
    _MODEL_NAME,
    _allocate_gpu_kv_cache,
    _build_token_ids,
    _get_chunk_size,
    _make_key,
    _poll_prefetch_status,
    _send_end_session,
    _send_lookup,
    _send_register_kv_cache,
    _send_unregister_kv_cache,
)
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey, KVCache
from lmcache.v1.multiprocess.futures import DeviceMessagingFuture, MessagingFuture
from lmcache.v1.multiprocess.group_view import EngineGroupInfo
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocols.base import RequestType
from lmcache.v1.platform.cuda.ipc_wrapper import CudaIPCWrapper

# Local
from .config import (
    GIB,
    MIB,
    L1BenchConfig,
    L1BenchPhaseResult,
    L1Operation,
)

_CHUNK_SIZE = 256
_BLOCK_SIZE = 16
_BLOCKS_PER_OBJECT = _CHUNK_SIZE // _BLOCK_SIZE
_KV_SIZE = 2
_NUM_LAYERS = 16
_HEAD_SIZE = 128
_BYTES_PER_OBJECT_HEAD = _NUM_LAYERS * _KV_SIZE * _CHUNK_SIZE * _HEAD_SIZE
_TRANSFER_TIMEOUT_SECONDS = 300.0
_WORKER_START_TIMEOUT_SECONDS = 30.0

# Logger callable type: takes a single string and prints / logs it.
LogFn = Callable[[str], None]

if _BYTES_PER_OBJECT_HEAD != MIB:
    raise AssertionError("synthetic KV geometry must produce 1 MiB per head")


def _submit_gpu_store(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    instance_id: int,
    block_ids: list[list[int]],
    event_ipc_handle: bytes,
    device: torch.device,
) -> DeviceMessagingFuture[bool]:
    """Submit one non-blocking GPU handle-path Store request."""
    raw_future: MessagingFuture[tuple[bytes, bool]] = client.submit_request(
        RequestType.STORE,
        [key, instance_id, block_ids, event_ipc_handle],
    )
    return raw_future.to_device_future(device=device)


def _submit_gpu_retrieve(
    client: MessageQueueClient,
    key: IPCCacheServerKey,
    instance_id: int,
    block_ids: list[list[int]],
    event_ipc_handle: bytes,
    device: torch.device,
) -> DeviceMessagingFuture[bool]:
    """Submit one non-blocking GPU handle-path Retrieve request."""
    raw_future: MessagingFuture[tuple[bytes, bool]] = client.submit_request(
        RequestType.RETRIEVE,
        [key, instance_id, block_ids, event_ipc_handle, 0],
    )
    return raw_future.to_device_future(device=device)


@dataclass
class _WorkerState:
    """One independent benchmark worker and its registered GPU cache."""

    worker_index: int
    instance_id: int
    client: MessageQueueClient
    kv_tensors: list[torch.Tensor]
    kv_wrappers: KVCache
    registered: bool = False


@dataclass
class _PendingRequest:
    """Resources retained until one Store or Load request completes."""

    worker_index: int
    request_id: str
    future: DeviceMessagingFuture[bool]
    producer_event: Any


class L1BenchRunner:
    """Drive benchmark workers with bounded in-flight request queues."""

    def __init__(
        self,
        config: L1BenchConfig,
        log: LogFn,
    ) -> None:
        """Create an uninitialized runner.

        Args:
            config: Validated workload configuration.
            log: Progress callback accepting one string.

        Raises:
            ValueError: If the configuration or CUDA device is invalid.
        """
        config.validate()
        self.config = config
        self.log = log
        self.device = torch.device(config.device)
        self.context: zmq.Context | None = None
        self.workers: list[_WorkerState] = []

    def run(self) -> tuple[L1BenchPhaseResult, L1BenchPhaseResult]:
        """Run Store warmup/measurement, then Load warmup/measurement.

        Returns:
            A ``(store_result, load_result)`` tuple.

        Raises:
            RuntimeError: If setup, lookup, Store, or Load fails.
        """
        try:
            self._initialize()

            if self.config.warmup_requests:
                self._prepare_requests("Store", 0, self.config.warmup_requests, 0)
                self._run_requests("Store", 0, self.config.warmup_requests)
            self._prepare_requests(
                "Store", self.config.warmup_requests, self.config.num_requests, 0
            )
            store_result = self._run_requests(
                "Store", self.config.warmup_requests, self.config.num_requests
            )

            if self.config.warmup_requests:
                self._prepare_requests(
                    "Load",
                    0,
                    self.config.warmup_requests,
                    self.config.num_keys,
                )
                self._run_requests("Load", 0, self.config.warmup_requests)
            self._prepare_requests(
                "Load",
                self.config.warmup_requests,
                self.config.num_requests,
                self.config.num_keys,
            )
            load_result = self._run_requests(
                "Load", self.config.warmup_requests, self.config.num_requests
            )
            return store_result, load_result
        finally:
            self._close()

    def _close(self) -> None:
        """Best-effort cleanup of registration, tensors, MQ client, and context."""
        for worker in self.workers:
            if worker.registered:
                try:
                    ok = _send_unregister_kv_cache(
                        worker.client,
                        instance_id=worker.instance_id,
                        use_handle=True,
                    )
                    if not ok:
                        self.log(
                            f"[Cleanup] worker {worker.worker_index} "
                            "unregister timed out"
                        )
                except Exception as exc:
                    self.log(
                        f"[Cleanup] worker {worker.worker_index} "
                        f"unregister failed: {exc}"
                    )
                worker.registered = False
            try:
                worker.client.close()
            except Exception as exc:
                self.log(
                    f"[Cleanup] worker {worker.worker_index} client close failed: {exc}"
                )
            worker.kv_wrappers.clear()
            worker.kv_tensors.clear()
        self.workers.clear()
        if self.context is not None:
            self.context.term()
            self.context = None
        if torch_dev.is_available():
            torch_dev.empty_cache()

    def _initialize(self) -> None:
        if self.device.type != "cuda" or not torch_dev.is_available():
            raise RuntimeError("L1 benchmark requires a CUDA device")
        torch_dev.set_device(self.device)
        self.context = zmq.Context()

        total_blocks_per_worker = (
            self.config.max_in_flight_requests
            * self.config.num_keys
            * _BLOCKS_PER_OBJECT
        )
        for worker_index in range(self.config.num_workers):
            worker = self._create_worker(
                worker_index,
                total_blocks_per_worker,
                check_chunk_size=worker_index == 0,
            )
            self.workers.append(worker)
            self._register(worker)

        # GPU tensor initialization is asynchronous. Finish it before the
        # request threads record producer events on their thread-local streams.
        torch_dev.synchronize(self.device)
        self.log(
            f"[Init] {self.config.num_workers} workers, "
            f"{self.config.max_in_flight_requests} max in-flight "
            "requests/worker, "
            f"{self.config.num_keys} keys/request, "
            f"{self.config.object_size_mib} MiB/object, "
            f"{self.config.gpu_buffer_bytes / GIB:.3f} GiB GPU buffer"
        )

    def _create_worker(
        self,
        worker_index: int,
        total_blocks: int,
        *,
        check_chunk_size: bool,
    ) -> _WorkerState:
        """Allocate one client and close it if setup fails before ownership."""
        if self.context is None:
            raise RuntimeError("ZMQ context is not initialized")

        client = MessageQueueClient(self.config.rpc_url, self.context)
        try:
            if check_chunk_size:
                chunk_size = _get_chunk_size(client)
                if chunk_size != _CHUNK_SIZE:
                    raise RuntimeError(
                        "L1 benchmark requires server --chunk-size "
                        f"{_CHUNK_SIZE}, got {chunk_size}"
                    )
            tensors = _allocate_gpu_kv_cache(
                num_layers=_NUM_LAYERS,
                num_heads=self.config.object_size_mib,
                head_size=_HEAD_SIZE,
                num_blocks=total_blocks,
                block_size=_BLOCK_SIZE,
                dtype=torch.uint8,
                device=self.device,
                kv_size=_KV_SIZE,
            )
            return _WorkerState(
                worker_index=worker_index,
                instance_id=(os.getpid() << 32) | worker_index,
                client=client,
                kv_tensors=tensors,
                kv_wrappers=[CudaIPCWrapper(tensor) for tensor in tensors],
            )
        except BaseException:
            with suppress(Exception):
                client.close()
            raise

    def _register(self, worker: _WorkerState) -> None:
        layout_hints = {
            "kv_layout": "NHD",
            "num_layers": _NUM_LAYERS,
            "num_heads": self.config.object_size_mib,
            "head_size": _HEAD_SIZE,
            "num_blocks": (
                self.config.max_in_flight_requests
                * self.config.num_keys
                * _BLOCKS_PER_OBJECT
            ),
            "block_size": _BLOCK_SIZE,
            "dtype": "uint8",
            "kv_size": _KV_SIZE,
        }
        groups = [
            EngineGroupInfo(
                engine_group_id=0,
                layer_indices=tuple(range(_NUM_LAYERS)),
                tokens_per_block=_BLOCK_SIZE,
            )
        ]
        result = _send_register_kv_cache(
            worker.client,
            instance_id=worker.instance_id,
            model_name=_MODEL_NAME,
            world_size=1,
            layout_hints=layout_hints,
            kv_caches=worker.kv_wrappers,
            use_gpu=True,
            use_handle=True,
            engine_group_infos=groups,
        )
        if not result:
            raise RuntimeError(
                f"worker {worker.worker_index} REGISTER_KV_CACHE failed; "
                "the server must use --supported-transfer-mode "
                "lmcache_driven or auto"
            )
        worker.registered = True

    def _run_requests(
        self,
        operation: L1Operation,
        first_request: int,
        request_count: int,
    ) -> L1BenchPhaseResult:
        """Run one Store or Load phase with independently paced workers."""
        start_time: float | None = None

        def record_start_time() -> None:
            nonlocal start_time
            start_time = time.perf_counter()

        start_barrier = threading.Barrier(
            len(self.workers) + 1,
            action=record_start_time,
            timeout=_WORKER_START_TIMEOUT_SECONDS,
        )
        with ThreadPoolExecutor(
            max_workers=len(self.workers),
            thread_name_prefix="l1-bench-worker",
        ) as executor:
            worker_futures = [
                executor.submit(
                    self._run_worker_requests,
                    worker,
                    operation,
                    first_request,
                    request_count,
                    start_barrier,
                )
                for worker in self.workers
            ]
            try:
                start_barrier.wait()
            except threading.BrokenBarrierError as exc:
                wait(worker_futures)
                for worker_future in worker_futures:
                    worker_error = worker_future.exception()
                    if worker_error is not None and not isinstance(
                        worker_error, threading.BrokenBarrierError
                    ):
                        raise RuntimeError(
                            "benchmark worker failed during startup"
                        ) from worker_error
                raise RuntimeError("benchmark workers failed to start") from exc

            wait(worker_futures)
            if start_time is None:
                raise RuntimeError("benchmark timer failed to start")
            elapsed = time.perf_counter() - start_time
            completed_requests = [
                completed_request
                for worker_future in worker_futures
                for completed_request in worker_future.result()
            ]

        for worker_index, request_id in completed_requests:
            _send_end_session(self.workers[worker_index].client, request_id)

        aggregate_request_count = request_count * self.config.num_workers
        return L1BenchPhaseResult(
            operation=operation,
            elapsed_seconds=elapsed,
            requests=aggregate_request_count,
            keys=aggregate_request_count * self.config.num_keys,
            bytes_transferred=(aggregate_request_count * self.config.bytes_per_request),
        )

    def _run_worker_requests(
        self,
        worker: _WorkerState,
        operation: L1Operation,
        first_request: int,
        request_count: int,
        start_barrier: threading.Barrier,
    ) -> list[tuple[int, str]]:
        """Run one worker's bounded request queue without cross-worker waits."""
        try:
            torch_dev.set_device(self.device)
            start_barrier.wait()
        except BaseException:
            start_barrier.abort()
            raise

        pending: deque[_PendingRequest] = deque()
        completed_requests: list[tuple[int, str]] = []
        for request_index in range(first_request, first_request + request_count):
            if len(pending) >= self.config.max_in_flight_requests:
                completed_requests.append(self._complete_request(pending.popleft()))

            slot = request_index % self.config.max_in_flight_requests
            pending.append(self._submit_request(worker, operation, request_index, slot))

        while pending:
            completed_requests.append(self._complete_request(pending.popleft()))
        return completed_requests

    def _submit_request(
        self,
        worker: _WorkerState,
        operation: L1Operation,
        request_index: int,
        slot: int,
    ) -> _PendingRequest:
        event = torch_dev.Event(interprocess=True)
        event.record()
        event_handle = event.ipc_handle()
        seq = self._sequence(request_index, worker.worker_index)
        request_id = f"l1-bench-{operation.lower()}-{seq}"
        token_ids = _build_token_ids(
            seq,
            self.config.num_keys * _CHUNK_SIZE - 1,
        )
        key = _make_key(
            token_ids,
            request_id,
            start=0,
            end=len(token_ids),
            worker_id=0,
            world_size=1,
        )
        block_ids = [self._block_ids(slot)]
        if operation == "Store":
            future = _submit_gpu_store(
                worker.client,
                key,
                worker.instance_id,
                block_ids,
                event_handle,
                self.device,
            )
        else:
            future = _submit_gpu_retrieve(
                worker.client,
                key,
                worker.instance_id,
                block_ids,
                event_handle,
                self.device,
            )

        return _PendingRequest(
            worker.worker_index,
            request_id,
            future,
            event,
        )

    def _complete_request(
        self,
        pending_request: _PendingRequest,
    ) -> tuple[int, str]:
        result = pending_request.future.result(timeout=_TRANSFER_TIMEOUT_SECONDS)
        if not result:
            raise RuntimeError(
                f"transfer failed for request {pending_request.request_id}"
            )
        return pending_request.worker_index, pending_request.request_id

    def _prepare_requests(
        self,
        operation: L1Operation,
        first_request: int,
        request_count: int,
        expected_hits: int,
    ) -> None:
        aggregate_requests = request_count * self.config.num_workers
        aggregate_keys = aggregate_requests * self.config.num_keys
        self.log(
            f"[{operation}] preparing {aggregate_requests} requests / "
            f"{aggregate_keys} keys"
        )
        for request_index in range(first_request, first_request + request_count):
            for worker in self.workers:
                seq = self._sequence(request_index, worker.worker_index)
                request_id = f"l1-bench-{operation.lower()}-{seq}"
                token_ids = _build_token_ids(
                    seq,
                    self.config.num_keys * _CHUNK_SIZE - 1,
                )
                key = _make_key(
                    token_ids,
                    request_id,
                    start=0,
                    end=len(token_ids),
                    worker_id=None,
                    world_size=1,
                )
                if not _send_lookup(worker.client, key, tp_size=1):
                    raise RuntimeError(f"LOOKUP failed for request {request_id}")
                hits = _poll_prefetch_status(worker.client, request_id)
                if hits != expected_hits:
                    raise RuntimeError(
                        f"LOOKUP found {hits}/{expected_hits} keys for "
                        f"request {request_id}"
                    )

    def _sequence(
        self,
        request_index: int,
        worker_index: int,
    ) -> int:
        return (
            self.config.key_offset
            + request_index * self.config.num_workers
            + worker_index
        )

    def _block_ids(self, slot: int) -> list[int]:
        blocks_per_request = self.config.num_keys * _BLOCKS_PER_OBJECT
        start = slot * blocks_per_request
        return list(range(start, start + blocks_per_request))
