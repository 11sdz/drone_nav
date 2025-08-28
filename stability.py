from collections import defaultdict, deque
from typing import Dict

class StabilityGate:
    """
    Tracks per-class presence & EMA confidence and applies hysteresis.
    Lock when score >= lock_thresh; unlock when score < unlock_thresh.
    """
    def __init__(self, history_len: int, presence_gamma: float, conf_alpha: float,
                 ema_beta: float, lock_thresh: float, unlock_thresh: float):
        self.history = defaultdict(lambda: deque(maxlen=history_len))  # type: Dict[str, deque[int]]
        self.ema_conf = defaultdict(lambda: 0.0)                       # type: Dict[str, float]
        self.locked = defaultdict(lambda: False)                       # type: Dict[str, bool]
        self.presence_gamma = presence_gamma
        self.conf_alpha = conf_alpha
        self.ema_beta = ema_beta
        self.lock_thresh = lock_thresh
        self.unlock_thresh = unlock_thresh

    def update(self, class_presence_now: Dict[str, int], class_conf_now: Dict[str, float]) -> Dict[str, float]:
        # Presence update (1 or 0 per class)
        seen = set(class_presence_now.keys())
        for name, val in class_presence_now.items():
            self.history[name].append(1 if val else 0)
        # also push 0 for classes not seen this frame
        for name in list(self.history.keys()):
            if name not in seen:
                self.history[name].append(0)

        # EMA confidence update
        for name, cf in class_conf_now.items():
            self.ema_conf[name] = (1 - self.ema_beta) * self.ema_conf[name] + self.ema_beta * cf

        # Score + hysteresis
        score = {}
        for name, buf in self.history.items():
            if len(buf) == 0:
                continue
            presence_ratio = sum(buf) / len(buf)
            s = (presence_ratio ** self.presence_gamma) * (max(self.ema_conf[name], 1e-3) ** self.conf_alpha)
            # Hysteresis
            if self.locked[name]:
                # remain locked unless confidence falls below unlock
                self.locked[name] = s >= self.unlock_thresh
            else:
                # acquire lock when high enough
                self.locked[name] = s >= self.lock_thresh
            score[name] = s
        return score

    def is_locked(self, class_name: str) -> bool:
        return self.locked[class_name]
