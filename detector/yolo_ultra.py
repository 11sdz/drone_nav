from typing import Iterable, List
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as yolo_colors
from core_types import IDetector, FrameDetections, BoxDet
from config import ModelCfg

class YoloUltranyxDetector(IDetector):
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        if cfg.device_index == 0 and not torch.cuda.is_available():
            raise RuntimeError("CUDA not available but device_index==0 requested.")
        self.model = YOLO(cfg.weights_path)

    def stream(self, video_path: str) -> Iterable[FrameDetections]:
        gen = self.model.predict(
            source=video_path,
            device=self.cfg.device_index if torch.cuda.is_available() else "cpu",
            stream=True,
            conf=self.cfg.conf,
            imgsz=self.cfg.img_size,
            half=(self.cfg.use_half and torch.cuda.is_available()),
            verbose=False
        )
        for res in gen:
            frame = res.orig_img
            names = res.names
            boxes: List[BoxDet] = []
            if res.boxes is not None and len(res.boxes) > 0:
                cls_ids = res.boxes.cls.int().tolist()
                confs = res.boxes.conf.detach().cpu().numpy().tolist()
                xyxy = res.boxes.xyxy.detach().cpu().numpy()
                for (bb, cid, cf) in zip(xyxy, cls_ids, confs):
                    boxes.append(BoxDet(xyxy=bb, cls_id=int(cid), conf=float(cf)))
            yield FrameDetections(frame_bgr=frame, boxes=boxes, names=names)
