# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench l1`` subcommand implementation.

This module provides argument registration via :func:`add_l1_arguments`
and the execution orchestrator :func:`run_l1_bench` for the GPU-to-L1
throughput benchmark.
"""

# Future
from __future__ import annotations

# Standard
from typing import TYPE_CHECKING
import argparse
import sys

# Local
from .config import (
    GIB,
    MIB,
    SUPPORTED_OBJECT_SIZE_MIB,
    L1BenchConfig,
    L1BenchPhaseResult,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.cli.commands.base import BaseCommand


# ---------------------------------------------------------------------------
# Parser registration
# ---------------------------------------------------------------------------


def add_l1_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``lmcache bench l1`` arguments to *parser*.

    Args:
        parser: The ``ArgumentParser`` for the L1 bench subcommand.
    """
    parser.add_argument(
        "--rpc-url",
        default="tcp://localhost:5555",
        help="ZMQ endpoint of the running MP server (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="CUDA device used by all benchmark workers (default: %(default)s).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Independent MP clients and CUDA-IPC contexts (default: "
            "%(default)s). Configure server --max-gpu-workers to at least "
            "this value."
        ),
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        default=4,
        help=(
            "LMCache keys transferred by each request; each key maps to one "
            "synthetic object (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--object-size-mib",
        type=int,
        choices=SUPPORTED_OBJECT_SIZE_MIB,
        default=1,
        help=(
            "Data size per synthetic object in MiB (default: %(default)s). "
            "The benchmark keeps chunk size 256 and scales the KV head count."
        ),
    )
    parser.add_argument(
        "--max-in-flight-requests",
        type=int,
        default=1,
        help=(
            "Maximum incomplete requests per worker (default: %(default)s). "
            "Each request owns a disjoint GPU block slot."
        ),
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help=("Measured requests issued by each worker (default: %(default)s)."),
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help=(
            "Unmeasured requests issued by each worker before each operation "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--key-offset",
        type=int,
        default=0,
        help="Starting synthetic key sequence offset (default: %(default)s).",
    )


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------


def run_l1_bench(command: "BaseCommand", args: argparse.Namespace) -> None:
    """Run the L1 throughput benchmark.

    Args:
        command: The owning :class:`BaseCommand` instance, used only
            to obtain a configured :class:`Metrics` object via
            ``command.create_metrics``.
        args: Parsed CLI arguments from the ``bench l1`` subparser.

    Raises:
        SystemExit: If configuration validation or benchmark execution fails.
    """
    config = L1BenchConfig(
        rpc_url=args.rpc_url,
        device=args.device,
        num_workers=args.num_workers,
        num_keys=args.num_keys,
        object_size_mib=args.object_size_mib,
        max_in_flight_requests=args.max_in_flight_requests,
        num_requests=args.num_requests,
        warmup_requests=args.warmup_requests,
        key_offset=args.key_offset,
    )
    try:
        config.validate()
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    # Lazy imports: keep CLI loadable without torch / native dependencies.
    # First Party
    from lmcache.cli.commands.bench.server_bench.helpers import (
        _require_full_install,
    )

    _require_full_install("lmcache bench l1")

    # Heavy CUDA / MP imports stay behind the full-install guard.
    # Local
    from .runner import L1BenchRunner

    quiet = getattr(args, "quiet", False)

    def log(message: str) -> None:
        if not quiet:
            print(message)

    try:
        store_result, load_result = L1BenchRunner(config, log).run()
    except Exception as exc:
        print(f"L1 benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    _emit_l1_metrics(command, args, config, store_result, load_result)


def _emit_l1_metrics(
    command: "BaseCommand",
    args: argparse.Namespace,
    config: L1BenchConfig,
    store_result: L1BenchPhaseResult,
    load_result: L1BenchPhaseResult,
) -> None:
    """Emit the L1 benchmark summary using the CLI metrics system.

    Args:
        command: Owning CLI command used to create metrics.
        args: Parsed CLI arguments controlling output handlers.
        config: Validated benchmark configuration.
        store_result: Completed Store phase result.
        load_result: Completed Load phase result.
    """
    metrics = command.create_metrics("L1 Throughput Benchmark Result", args)
    cfg = metrics.add_section("config", "Configuration")
    cfg.add("transfer_mode", "Transfer mode", "lmcache_driven")
    cfg.add("rpc_url", "RPC URL", config.rpc_url)
    cfg.add("device", "CUDA device", config.device)
    cfg.add("num_workers", "Workers", config.num_workers)
    cfg.add("num_keys", "Keys / request", config.num_keys)
    cfg.add("object_size_mib", "Object size (MiB)", config.object_size_mib)
    cfg.add("object_size_bytes", "Object size (bytes)", config.object_size_bytes)
    cfg.add(
        "data_per_request_mib",
        "Data / request (MiB)",
        round(config.bytes_per_request / MIB, 3),
    )
    cfg.add(
        "max_in_flight_requests_per_worker",
        "Max in-flight requests / worker",
        config.max_in_flight_requests,
    )
    cfg.add(
        "max_in_flight_requests_total",
        "Max in-flight requests total",
        config.max_in_flight_requests_total,
    )
    cfg.add(
        "max_in_flight_keys_total",
        "Max in-flight keys total",
        config.max_in_flight_keys_total,
    )
    cfg.add(
        "num_requests_per_worker",
        "Measured requests / worker",
        config.num_requests,
    )
    cfg.add(
        "warmup_requests_per_worker",
        "Warmup requests / worker",
        config.warmup_requests,
    )
    cfg.add(
        "measured_requests_total",
        "Measured requests total",
        config.measured_requests,
    )
    cfg.add(
        "measured_keys_total",
        "Measured keys total",
        config.measured_keys,
    )
    cfg.add("key_offset", "Key sequence offset", config.key_offset)
    cfg.add(
        "measured_data_gib",
        "Measured data / phase (GiB)",
        round(config.measured_bytes / GIB, 3),
    )
    cfg.add(
        "resident_data_gib",
        "Resident data after Store (GiB)",
        round(config.resident_bytes / GIB, 3),
    )
    cfg.add(
        "gpu_buffer_gib",
        "Client GPU buffer (GiB)",
        round(config.gpu_buffer_bytes / GIB, 3),
    )
    cfg.add("required_l1_gib", "Recommended L1 size (GiB)", config.required_l1_gib)

    for section_id, result in (("store", store_result), ("load", load_result)):
        section = metrics.add_section(section_id, result.operation)
        section.add("requests", "Successful requests", result.requests)
        section.add("keys", "Successful keys", result.keys)
        section.add(
            "data_gib",
            "Transferred data (GiB)",
            round(result.bytes_transferred / GIB, 3),
        )
        section.add(
            "elapsed_seconds",
            "Phase duration (s)",
            round(result.elapsed_seconds, 6),
        )
        section.add(
            "throughput_gbps",
            "Throughput (GB/s)",
            round(result.throughput_gbps, 3),
        )
        section.add(
            "throughput_gibps",
            "Throughput (GiB/s)",
            round(result.throughput_gibps, 3),
        )
        section.add(
            "keys_per_second",
            "Keys / second",
            round(result.keys_per_second, 3),
        )

    metrics.emit()
