"""krea2 API schema, generated from the shared contract."""

import sys
from pathlib import Path

workers_dir = Path(__file__).resolve().parents[1]
if (workers_dir / "shared").is_dir():
    sys.path.insert(0, str(workers_dir))
from shared.api import KREA_API

INPUT_SCHEMA = KREA_API.input_schema()
