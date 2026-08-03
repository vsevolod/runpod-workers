"""RunPod input validation schema for lora_downloader (job-level only)."""

from download import MAX_ITEMS

INPUT_SCHEMA = {
    "items": {
        "type": list,
        "required": True,
        "constraints": lambda items: (
            isinstance(items, list) and 1 <= len(items) <= MAX_ITEMS
        ),
    },
}
