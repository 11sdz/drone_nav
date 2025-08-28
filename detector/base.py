from typing import Iterable
from core_types import IDetector, FrameDetections

class DetectorBase(IDetector):
    def stream(self, video_path: str) -> Iterable[FrameDetections]:
        raise NotImplementedError
