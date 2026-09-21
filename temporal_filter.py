from collections import deque

import numpy as np

from labels import LABELS_EN


class TemporalScoreFilter:
    """
    Sliding temporal window.

    Decision = argmax(sum(scores of last N processed frames))
    """

    def __init__(self, window_size=10):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def reset(self):
        self.history.clear()

    @property
    def ready(self):
        return (len(self.history) == self.window_size)

    def update(self, scores):
        self.history.append(np.asarray(scores).copy())

        if not self.ready:
            return {
                "ready": False,
                "window_fill": len(self.history),
                "window_size": (self.window_size),
                "class_id": None,
                "label": "Warming up",
                "score_sum": None,
            }

        accumulated_scores = np.sum(np.stack(self.history), axis=0)

        class_id = int(np.argmax(accumulated_scores))

        return {
            "ready": True,
            "window_fill": len(self.history),
            "window_size": (self.window_size),
            "class_id": class_id,
            "label": (LABELS_EN[class_id]),
            "score_sum": float(accumulated_scores[class_id]),
        }
