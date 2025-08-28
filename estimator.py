from typing import Dict, List, Tuple, Optional
import numpy as np
from geo import latlon_to_xy_m, xy_m_to_latlon, haversine_m
from core_types import LatLon

class PositionEstimator:
    def __init__(self, lat0: float, lon0: float, pos_alpha: float):
        self.lat0 = lat0
        self.lon0 = lon0
        self.pos_alpha = pos_alpha
        self.pred_s: Optional[LatLon] = None
        self.prev_s: Optional[LatLon] = None

    def estimate(self,
                 class_weights: List[float],
                 class_locs: List[LatLon],
                 dt: float) -> Tuple[Optional[LatLon], float, float]:
        """
        Returns: (smoothed_position, speed_mps, speed_kmh)
        """
        if not class_weights:
            return self.pred_s, 0.0, 0.0

        w = np.asarray(class_weights, dtype=float)
        w = w / (w.sum() + 1e-9)

        xs, ys = [], []
        for w_i, ll in zip(w, class_locs):
            x, y = latlon_to_xy_m(ll.lat, ll.lon, self.lat0, self.lon0)
            xs.append(w_i * x); ys.append(w_i * y)
        xw, yw = float(np.sum(xs)), float(np.sum(ys))
        lat, lon = xy_m_to_latlon(xw, yw, self.lat0, self.lon0)
        pred = LatLon(lat, lon)

        if self.pred_s is None:
            self.pred_s = pred
        else:
            self.pred_s = LatLon(
                lat=self.pred_s.lat * (1 - self.pos_alpha) + pred.lat * self.pos_alpha,
                lon=self.pred_s.lon * (1 - self.pos_alpha) + pred.lon * self.pos_alpha
            )

        speed_mps = 0.0
        if self.prev_s is not None and dt > 0:
            d_m = haversine_m(self.prev_s.lat, self.prev_s.lon, self.pred_s.lat, self.pred_s.lon)
            speed_mps = d_m / dt
        self.prev_s = self.pred_s
        return self.pred_s, speed_mps, speed_mps * 3.6
