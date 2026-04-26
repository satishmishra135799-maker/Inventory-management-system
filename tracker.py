"""
tracker.py — DeepSORT-based product tracker.
Counts each physical product exactly once (not per-frame).
Uses IoU-only tracking with dummy embeddings — no extra torch dependency.
"""

import sys
import types
import numpy as np

# Python 3.13 compatibility patch
if 'pkg_resources' not in sys.modules:
    _pkg = types.ModuleType('pkg_resources')
    _pkg.require = lambda *a, **k: None
    sys.modules['pkg_resources'] = _pkg

from deep_sort_realtime.deepsort_tracker import DeepSort

MAX_AGE = 30
N_INIT  = 3
EMBED_DIM = 128  # dummy embedding size


class ProductTracker:
    def __init__(self):
        self._init_tracker()
        self.counted: dict[int, str] = {}   # track_id → label (counted once)
        self.cart: dict[str, int] = {}       # label → count
        self.pending: list[str] = []         # brands awaiting variant selection

    def _init_tracker(self):
        self.tracker = DeepSort(
            max_age=MAX_AGE,
            n_init=N_INIT,
            embedder=None,
            embedder_gpu=False,
        )

    def update(self, detections: list[dict], frame) -> dict[str, int]:
        """
        detections: list of {'bbox': [x1,y1,x2,y2], 'conf': float, 'label': str}
        Returns current cart.
        """
        if not detections:
            self.tracker.update_tracks([], embeds=[], frame=frame)
            return self.cart

        ds_input = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            ds_input.append(([x1, y1, x2 - x1, y2 - y1], d['conf'], d['label']))

        # Dummy unit embeddings — IoU handles association
        embeds = [np.ones(EMBED_DIM) for _ in ds_input]

        tracks = self.tracker.update_tracks(ds_input, embeds=embeds, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            tid   = track.track_id
            label = track.get_det_class()
            if label is None or tid in self.counted:
                continue
            self.counted[tid] = label
            self.pending.append(label)   # cashier picks variant before cart add

        return self.cart

    def reset(self):
        self._init_tracker()
        self.counted.clear()
        self.cart.clear()
        self.pending.clear()

    def remove_item(self, label: str) -> bool:
        if self.cart.get(label, 0) > 0:
            self.cart[label] -= 1
            if self.cart[label] == 0:
                del self.cart[label]
            return True
        return False
