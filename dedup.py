from typing import List, Dict, Tuple
import numpy as np
from core_types import BoxDet

def deduplicate_by_class(
    boxes: List[BoxDet],
    id_to_name: Dict[int, str],
    passthrough_name: str = "other"
) -> Tuple[List[BoxDet], Dict[str, float]]:
    """
    Keep only best (highest conf) per class, except keep all for 'other'.
    Returns kept boxes and class->conf dict.
    """
    best: Dict[int, Tuple[int, float]] = {}
    keep_idx = []

    for i, b in enumerate(boxes):
        cname = id_to_name[int(b.cls_id)].lower()
        if cname == passthrough_name:
            keep_idx.append(i)
        else:
            cur = best.get(b.cls_id)
            if cur is None or b.conf > cur[1]:
                best[b.cls_id] = (i, b.conf)

    keep_idx.extend(i for i, _ in best.values())
    keep_idx = sorted(set(keep_idx))
    kept = [boxes[i] for i in keep_idx]

    class_conf = {}
    for b in kept:
        cname = id_to_name[int(b.cls_id)]
        class_conf[cname] = max(class_conf.get(cname, 0.0), b.conf)
    return kept, class_conf
