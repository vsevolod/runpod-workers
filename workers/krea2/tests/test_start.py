"""Entrypoint concurrency contract for this worker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from shared.tests import startup_contract


class StartTests(startup_contract.StartupContractTests):
    start = Path(__file__).parents[1] / "start.sh"
    profile = "krea2"
