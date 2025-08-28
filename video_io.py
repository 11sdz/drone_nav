import cv2
from typing import Tuple
import numpy as np
from core_types import IVideoReader, IVideoWriter

class Cv2VideoReader(IVideoReader):
    def __init__(self):
        self.cap = None

    def open(self, path: str) -> Tuple[int, int, float]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.cap = cap
        return w, h, fps

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

class Cv2VideoWriter(IVideoWriter):
    def __init__(self, path: str, fps: float, size: Tuple[int, int]):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)

    def write(self, frame_bgr: np.ndarray) -> None:
        self.writer.write(frame_bgr)

    def close(self) -> None:
        self.writer.release()
