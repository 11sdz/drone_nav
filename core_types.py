from dataclasses import dataclass
from typing import Protocol, Iterable, Tuple, Dict, List, Optional
import numpy as np

@dataclass(frozen=True)
class LatLon:
    lat: float
    lon: float

@dataclass
class BoxDet:
    xyxy: np.ndarray  # shape (4,)
    cls_id: int
    conf: float

@dataclass
class FrameDetections:
    frame_bgr: np.ndarray
    boxes: List[BoxDet]
    names: Dict[int, str]  # id -> name

class IDetector(Protocol):
    def stream(self, video_path: str) -> Iterable[FrameDetections]: ...

class IVideoWriter(Protocol):
    def write(self, frame_bgr: np.ndarray) -> None: ...
    def close(self) -> None: ...

class IVideoReader(Protocol):
    def open(self, path: str) -> Tuple[int, int, float]: ...  # (W, H, fps)
    def close(self) -> None: ...

class ITileProvider(Protocol):
    def get_tile(self, z: int, x: int, y: int) -> Optional[np.ndarray]: ...

class IHUDRenderer(Protocol):
    def render(self, center: LatLon, path: List[LatLon], nodes: List[Tuple[str, LatLon]], actual: Optional[LatLon] = None) -> np.ndarray: ...

# Strategy protocols for SOLID/OOP pluggability
class IPositionPredictor(Protocol):
    def predict(self, class_weights: List[float], class_locs: List[LatLon]) -> Optional[LatLon]: ...

class IPositionSmoother(Protocol):
    def update(self, position: Optional[LatLon], dt: float) -> Optional[LatLon]: ...

class ISpeedEstimator(Protocol):
    def update(self, position: Optional[LatLon], dt: float) -> Tuple[float, float]: ...