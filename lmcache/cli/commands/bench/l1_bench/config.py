# SPDX-License-Identifier: Apache-2.0
"""Configuration and phase results for ``lmcache bench l1``."""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
from typing import Literal
import math

MIB = 1 << 20
GIB = 1 << 30
SUPPORTED_OBJECT_SIZE_MIB = (1, 4, 8, 16, 32)
L1Operation = Literal["Store", "Load"]


@dataclass(frozen=True)
class L1BenchConfig:
    """Validated configuration for the GPU-to-L1 throughput benchmark.

    Each benchmark worker issues ``num_requests`` measured requests, each
    carrying ``num_keys`` objects. At most ``max_in_flight_requests`` requests
    per worker may remain incomplete before the oldest request is awaited.
    """

    rpc_url: str
    device: str
    num_workers: int
    num_keys: int
    object_size_mib: int
    max_in_flight_requests: int
    num_requests: int
    warmup_requests: int
    key_offset: int

    def validate(self) -> None:
        """Validate the supported GPU-to-L1 workload.

        Raises:
            ValueError: If an option is outside its supported range.
        """
        if not self.rpc_url:
            raise ValueError("rpc_url must not be empty")
        device_parts = self.device.split(":", maxsplit=1)
        if device_parts[0] != "cuda" or (
            len(device_parts) == 2 and not device_parts[1].isdigit()
        ):
            raise ValueError("device must be 'cuda' or 'cuda:<index>'")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.num_keys <= 0:
            raise ValueError("num_keys must be positive")
        if self.object_size_mib not in SUPPORTED_OBJECT_SIZE_MIB:
            raise ValueError(
                f"object_size_mib must be one of {list(SUPPORTED_OBJECT_SIZE_MIB)}"
            )
        if self.max_in_flight_requests <= 0:
            raise ValueError("max_in_flight_requests must be positive")
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if self.warmup_requests < 0:
            raise ValueError("warmup_requests must be non-negative")
        if self.key_offset < 0:
            raise ValueError("key_offset must be non-negative")

        max_seq = (
            self.key_offset
            + (self.warmup_requests + self.num_requests) * self.num_workers
            - 1
        )
        if max_seq >= 1 << 64:
            raise ValueError("request sequence range exceeds uint64")

    @property
    def object_size_bytes(self) -> int:
        """Return bytes in one synthetic LMCache object."""
        return self.object_size_mib * MIB

    @property
    def bytes_per_request(self) -> int:
        """Return payload bytes carried by one request."""
        return self.num_keys * self.object_size_bytes

    @property
    def measured_requests(self) -> int:
        """Return aggregate measured requests across all workers."""
        return self.num_workers * self.num_requests

    @property
    def measured_keys(self) -> int:
        """Return aggregate measured keys across all workers."""
        return self.measured_requests * self.num_keys

    @property
    def measured_bytes(self) -> int:
        """Return measured payload bytes in one Store or Load phase."""
        return self.measured_requests * self.bytes_per_request

    @property
    def resident_bytes(self) -> int:
        """Return bytes resident after all warmup and measured Store requests."""
        requests_per_worker = self.warmup_requests + self.num_requests
        return self.num_workers * requests_per_worker * self.bytes_per_request

    @property
    def max_in_flight_requests_total(self) -> int:
        """Return the aggregate maximum number of incomplete requests."""
        return self.num_workers * self.max_in_flight_requests

    @property
    def max_in_flight_keys_total(self) -> int:
        """Return the aggregate maximum number of incomplete keys."""
        return self.max_in_flight_requests_total * self.num_keys

    @property
    def gpu_buffer_bytes(self) -> int:
        """Return bytes held by the paged-KV block ring."""
        return self.max_in_flight_requests_total * self.bytes_per_request

    @property
    def required_l1_gib(self) -> int:
        """Return integer L1 capacity with one GiB of headroom."""
        return math.ceil(self.resident_bytes / GIB) + 1


@dataclass(frozen=True)
class L1BenchPhaseResult:
    """Aggregate measured result for one Store or Load operation.

    ``elapsed_seconds`` spans submission of the first measured request through
    completion of every device-aware future. Lookup preparation, warmup, and
    session cleanup are outside this interval.
    """

    operation: L1Operation
    elapsed_seconds: float
    requests: int
    keys: int
    bytes_transferred: int

    @property
    def throughput_gbps(self) -> float:
        """Return aggregate decimal GB/s."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.bytes_transferred / self.elapsed_seconds / 1e9

    @property
    def throughput_gibps(self) -> float:
        """Return aggregate binary GiB/s."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.bytes_transferred / self.elapsed_seconds / GIB

    @property
    def keys_per_second(self) -> float:
        """Return completed keys per second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.keys / self.elapsed_seconds
