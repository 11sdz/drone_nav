from typing import List, Tuple, Optional
import numpy as np
from geo import latlon_to_xy_m, xy_m_to_latlon, haversine_m
from core_types import LatLon, IPositionPredictor, IPositionSmoother, ISpeedEstimator


class WeightedBarycenterPredictor(IPositionPredictor):
    def __init__(self, lat0: float, lon0: float):
        self.lat0 = lat0
        self.lon0 = lon0

    def predict(self, class_weights: List[float], class_locs: List[LatLon]) -> Optional[LatLon]:
        if not class_weights:
            return None
        w = np.asarray(class_weights, dtype=float)
        w = w / (w.sum() + 1e-9)
        xs, ys = [], []
        for w_i, ll in zip(w, class_locs):
            x, y = latlon_to_xy_m(ll.lat, ll.lon, self.lat0, self.lon0)
            xs.append(w_i * x)
            ys.append(w_i * y)
        xw, yw = float(np.sum(xs)), float(np.sum(ys))
        lat, lon = xy_m_to_latlon(xw, yw, self.lat0, self.lon0)
        return LatLon(lat, lon)


class EmaPositionSmoother(IPositionSmoother):
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.state: Optional[LatLon] = None

    def update(self, position: Optional[LatLon], dt: float) -> Optional[LatLon]:
        if position is None:
            return self.state
        if self.state is None:
            self.state = position
            return self.state
        self.state = LatLon(
            lat=self.state.lat * (1 - self.alpha) + position.lat * self.alpha,
            lon=self.state.lon * (1 - self.alpha) + position.lon * self.alpha,
        )
        return self.state


class DisplacementSpeedEstimator(ISpeedEstimator):
    def __init__(self, speed_alpha: float = 0.0):
        self.prev: Optional[LatLon] = None
        self.speed_alpha = speed_alpha
        self.speed_state: float = 0.0

    def update(self, position: Optional[LatLon], dt: float) -> Tuple[float, float]:
        if position is None or dt <= 0:
            return self.speed_state, self.speed_state * 3.6
        speed_mps = 0.0
        if self.prev is not None:
            d_m = haversine_m(self.prev.lat, self.prev.lon, position.lat, position.lon)
            speed_mps = d_m / dt
        self.prev = position
        if self.speed_alpha > 0:
            self.speed_state = (1 - self.speed_alpha) * self.speed_state + self.speed_alpha * speed_mps
        else:
            self.speed_state = speed_mps
        return self.speed_state, self.speed_state * 3.6


class PositionEstimator:
    """
    Composes strategies for prediction, smoothing, and speed.
    """
    def __init__(self, predictor: IPositionPredictor, smoother: IPositionSmoother, speed: ISpeedEstimator):
        self.predictor = predictor
        self.smoother = smoother
        self.speed = speed

    def estimate(self, class_weights: List[float], class_locs: List[LatLon], dt: float) -> Tuple[Optional[LatLon], float, float]:
        pred = self.predictor.predict(class_weights, class_locs)
        smoothed = self.smoother.update(pred, dt)
        spd_mps, spd_kmh = self.speed.update(smoothed, dt)
        return smoothed, spd_mps, spd_kmh


class KalmanCvSmoother(IPositionSmoother):
    """
    Constant-velocity Kalman smoother in local meters (x,y,vx,vy).
    Converts LatLon<->meters using fixed origin defined by predictor's reference (passed at init).
    """
    def __init__(self, lat0: float, lon0: float, q_pos: float = 1.0, q_vel: float = 1.0, r_pos: float = 9.0):
        from geo import latlon_to_xy_m, xy_m_to_latlon  # local import to avoid cycles at top
        self.lat0 = lat0
        self.lon0 = lon0
        self._to_xy = latlon_to_xy_m
        self._to_ll = xy_m_to_latlon
        # state x = [x, y, vx, vy]
        self.x = None  # shape (4,1)
        self.P = None  # shape (4,4)
        # process/measurement noise (tuned constants; can be config-driven later)
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r_pos = r_pos

    def update(self, position: Optional[LatLon], dt: float) -> Optional[LatLon]:
        import numpy as np
        if dt <= 0:
            dt = 1e-3
        # Build motion model
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)
        G = np.array([[0.5*dt*dt, 0],
                      [0, 0.5*dt*dt],
                      [dt, 0],
                      [0, dt]], dtype=float)
        Q = G @ np.diag([self.q_pos, self.q_pos]) @ G.T + np.diag([0,0,self.q_vel,self.q_vel]) * 1e-6
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=float)
        R = np.eye(2) * self.r_pos

        # Predict step
        if self.x is not None:
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q

        # Update with measurement (if available)
        if position is not None:
            zx, zy = self._to_xy(position.lat, position.lon, self.lat0, self.lon0)
            z = np.array([[zx], [zy]])
            if self.x is None:
                # init state from first position
                self.x = np.array([[zx],[zy],[0.0],[0.0]])
                self.P = np.eye(4) * 100.0
            else:
                S = H @ self.P @ H.T + R
                K = self.P @ H.T @ np.linalg.inv(S)
                y = z - (H @ self.x)
                self.x = self.x + K @ y
                self.P = (np.eye(4) - K @ H) @ self.P

        if self.x is None:
            return None
        x, y = float(self.x[0,0]), float(self.x[1,0])
        lat, lon = self._to_ll(x, y, self.lat0, self.lon0)
        return LatLon(lat, lon)
