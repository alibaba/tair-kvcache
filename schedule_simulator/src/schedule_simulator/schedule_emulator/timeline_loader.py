"""Timeline JSONL file loader with stable pod name to ID mapping."""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TimelineLoader:
    """Load a timeline JSONL file and build a stable pod-name-to-ID mapping.

    The mapping is built by sorting all unique pod names alphabetically
    and assigning consecutive integer IDs starting from 0.
    This ensures the mapping is deterministic regardless of file order.

    Note: Only pod names are stored in memory (not full records),
    so this is safe for multi-GB files.
    """

    def __init__(self, timeline_file: str, pod_prefix: str = None):
        self._timeline_file = timeline_file
        self._pod_prefix = pod_prefix
        self._pod_names_sorted: List[str] = []
        self._pod_name_to_id: Dict[str, int] = {}
        self._scan_pods()

    def _scan_pods(self):
        """Scan the JSONL file to collect all unique pod names."""
        pod_names = set()
        line_count = 0
        with open(self._timeline_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pods = record.get("pods", [])
                if pods:
                    if self._pod_prefix:
                        pods = [p for p in pods if p.startswith(self._pod_prefix)]
                    pod_names.update(pods)
                line_count += 1

        # Stable sorted mapping
        self._pod_names_sorted = sorted(pod_names)
        self._pod_name_to_id = {
            name: idx for idx, name in enumerate(self._pod_names_sorted)
        }
        logger.info(
            f"TimelineLoader: scanned {line_count} records, "
            f"found {len(self._pod_names_sorted)} unique pods"
        )

    @property
    def num_pods(self) -> int:
        """Number of unique pods in the timeline file."""
        return len(self._pod_names_sorted)

    @property
    def pod_name_to_id(self) -> Dict[str, int]:
        """Mapping from pod name to sequential integer ID."""
        return dict(self._pod_name_to_id)

    @property
    def pod_names(self) -> List[str]:
        """Sorted list of unique pod names."""
        return list(self._pod_names_sorted)

    def get_pod_index(self, pod_name: str) -> Optional[int]:
        """Get the integer ID for a given pod name."""
        return self._pod_name_to_id.get(pod_name)
