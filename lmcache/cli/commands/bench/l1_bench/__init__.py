# SPDX-License-Identifier: Apache-2.0
"""``lmcache bench l1`` subpackage.

Exposes :class:`L1BenchCommand` for auto-discovery by
:class:`~lmcache.cli.commands.base.CompositeCommand`.
"""

# Standard
import argparse

# First Party
from lmcache.cli.commands.base import BaseCommand


class L1BenchCommand(BaseCommand):
    """Benchmark GPU-to-L1 Store / Load throughput through the MP server."""

    def name(self) -> str:
        """Return the benchmark target name."""
        return "l1"

    def help(self) -> str:
        """Return the short CLI help text."""
        return "Benchmark GPU-to-L1 throughput through the MP cache server."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add ``lmcache bench l1`` arguments to *parser*.

        Args:
            parser: The ``ArgumentParser`` for the L1 bench subcommand.
        """
        # First Party
        from lmcache.cli.commands.bench.l1_bench.command import (
            add_l1_arguments,
        )

        add_l1_arguments(parser)

    def execute(self, args: argparse.Namespace) -> None:
        """Run the L1 throughput benchmark.

        Args:
            args: Parsed CLI arguments from the ``bench l1`` subparser.
        """
        # First Party
        from lmcache.cli.commands.bench.l1_bench.command import run_l1_bench

        run_l1_bench(self, args)
