from typing import List, Tuple, Optional, Dict
import numpy as np
from collections import deque
from dataclasses import dataclass
from core_types import LatLon, IPositionPredictor
from geo import latlon_to_xy_m, xy_m_to_latlon, haversine_m, offset_latlon_by_m
from flow import OpticalFlowDetector, FlowConfig

@dataclass
class RobustPredictorConfig:
    # YOLO-based prediction weights
    yolo_confidence_threshold: float = 0.6
    yolo_stability_threshold: float = 0.7
    
    # Optical flow settings
    flow_confidence_threshold: float = 0.4
    flow_stability_threshold: float = 0.5
    flow_primary_weight: float = 0.8  # Weight when flow is primary
    
    # Noise filtering
    max_jump_distance_m: float = 100.0
    min_detection_confidence: float = 0.3
    outlier_rejection_enabled: bool = True
    
    # History and smoothing
    history_length: int = 10
    position_alpha: float = 0.3
    velocity_alpha: float = 0.2
    
    # Switching logic
    min_yolo_detections: int = 1
    flow_fallback_enabled: bool = True
    hybrid_mode_enabled: bool = True

class RobustPositionPredictor(IPositionPredictor):
    """
    Robust position predictor that switches between YOLO detections and optical flow
    based on confidence and stability. Includes noise filtering and history tracking.
    """
    
    def __init__(self, lat0: float, lon0: float, config: RobustPredictorConfig):
        self.lat0 = lat0
        self.lon0 = lon0
        self.config = config
        
        # Initialize optical flow detector
        flow_config = FlowConfig(
            max_features=500,
            quality_level=0.01,
            min_distance=10,
            confidence_threshold=config.flow_confidence_threshold,
            history_length=config.history_length,
            velocity_alpha=config.velocity_alpha
        )
        self.flow_detector = OpticalFlowDetector(flow_config)
        
        # History tracking
        self.position_history = deque(maxlen=config.history_length)
        self.velocity_history = deque(maxlen=config.history_length)
        self.confidence_history = deque(maxlen=config.history_length)
        
        # State tracking
        self.last_position: Optional[LatLon] = None
        self.last_velocity = (0.0, 0.0)  # (east, north)
        self.last_dt: float = 1.0
        self.current_mode = "yolo"  # "yolo", "flow", "hybrid"
        self.mode_confidence = 0.0
        
        # Noise filtering
        self.outlier_rejection_enabled = config.outlier_rejection_enabled
        self.max_jump_distance = config.max_jump_distance_m
        
    def predict(self, class_weights: List[float], class_locs: List[LatLon]) -> Optional[LatLon]:
        """
        Predict position using YOLO detections, optical flow, or hybrid approach
        based on confidence and stability.
        """
        # Get YOLO-based prediction
        yolo_pred, yolo_confidence = self._predict_yolo(class_weights, class_locs)
        
        # Get optical flow prediction
        flow_pred, flow_confidence = self._predict_flow()
        
        # Determine prediction mode and combine results
        prediction = self._combine_predictions(yolo_pred, yolo_confidence, flow_pred, flow_confidence)
        
        # Update history and state
        if prediction is not None:
            self._update_history(prediction, yolo_confidence, flow_confidence)
            self.last_position = prediction
            
        return prediction
    
    def _predict_yolo(self, class_weights: List[float], class_locs: List[LatLon]) -> Tuple[Optional[LatLon], float]:
        """Predict position using YOLO detections with noise filtering"""
        if not class_weights or len(class_weights) < self.config.min_yolo_detections:
            return None, 0.0
            
        # Calculate weighted barycenter
        w = np.asarray(class_weights, dtype=float)
        w = w / (w.sum() + 1e-9)
        
        xs, ys = [], []
        for w_i, ll in zip(w, class_locs):
            x, y = latlon_to_xy_m(ll.lat, ll.lon, self.lat0, self.lon0)
            xs.append(w_i * x)
            ys.append(w_i * y)
            
        xw, yw = float(np.sum(xs)), float(np.sum(ys))
        lat, lon = xy_m_to_latlon(xw, yw, self.lat0, self.lon0)
        yolo_pred = LatLon(lat, lon)
        
        # Calculate confidence based on detection quality
        yolo_confidence = self._calculate_yolo_confidence(class_weights, class_locs)
        
        # Apply noise filtering
        if self.outlier_rejection_enabled and self.last_position is not None:
            if not self._is_prediction_valid(yolo_pred, yolo_confidence):
                return None, 0.0
                
        return yolo_pred, yolo_confidence
    
    def _predict_flow(self) -> Tuple[Optional[LatLon], float]:
        """Predict position using optical flow"""
        if self.last_position is None:
            return None, 0.0
            
        flow_confidence = self.flow_detector.get_flow_confidence()
        flow_stable = self.flow_detector.is_flow_stable()
        
        if flow_confidence < self.config.flow_confidence_threshold or not flow_stable:
            return None, 0.0
            
        # Predict position using flow
        flow_pred = self.flow_detector.predict_position(self.last_position, self.last_dt)
        
        return flow_pred, flow_confidence
    
    def _combine_predictions(self, yolo_pred: Optional[LatLon], yolo_confidence: float,
                           flow_pred: Optional[LatLon], flow_confidence: float) -> Optional[LatLon]:
        """Combine YOLO and flow predictions based on confidence and mode"""
        
        # Determine prediction mode
        mode = self._determine_prediction_mode(yolo_pred, yolo_confidence, flow_pred, flow_confidence)
        self.current_mode = mode
        
        if mode == "yolo":
            return yolo_pred
        elif mode == "flow":
            return flow_pred
        elif mode == "hybrid" and yolo_pred is not None and flow_pred is not None:
            return self._hybrid_prediction(yolo_pred, yolo_confidence, flow_pred, flow_confidence)
        else:
            # Fallback to best available
            if yolo_pred is not None and yolo_confidence > flow_confidence:
                return yolo_pred
            elif flow_pred is not None:
                return flow_pred
            else:
                return yolo_pred  # Last resort
    
    def _determine_prediction_mode(self, yolo_pred: Optional[LatLon], yolo_confidence: float,
                                 flow_pred: Optional[LatLon], flow_confidence: float) -> str:
        """Determine which prediction mode to use based on confidence and stability"""
        
        yolo_available = yolo_pred is not None and yolo_confidence >= self.config.yolo_confidence_threshold
        flow_available = flow_pred is not None and flow_confidence >= self.config.flow_confidence_threshold
        
        if not yolo_available and not flow_available:
            return "yolo"  # Default fallback
            
        if not yolo_available and flow_available:
            return "flow"
            
        if yolo_available and not flow_available:
            return "yolo"
            
        # Both available - choose based on confidence and stability
        if self.config.hybrid_mode_enabled:
            return "hybrid"
        elif yolo_confidence > flow_confidence:
            return "yolo"
        else:
            return "flow"
    
    def _hybrid_prediction(self, yolo_pred: LatLon, yolo_confidence: float,
                          flow_pred: LatLon, flow_confidence: float) -> LatLon:
        """Combine YOLO and flow predictions using weighted average"""
        
        # Calculate weights based on confidence
        total_confidence = yolo_confidence + flow_confidence
        if total_confidence < 1e-6:
            return yolo_pred
            
        yolo_weight = yolo_confidence / total_confidence
        flow_weight = flow_confidence / total_confidence
        
        # Apply flow primary weight if flow is very confident
        if flow_confidence >= self.config.flow_stability_threshold:
            flow_weight *= self.config.flow_primary_weight
            yolo_weight = 1.0 - flow_weight
        
        # Weighted average in meters
        yolo_x, yolo_y = latlon_to_xy_m(yolo_pred.lat, yolo_pred.lon, self.lat0, self.lon0)
        flow_x, flow_y = latlon_to_xy_m(flow_pred.lat, flow_pred.lon, self.lat0, self.lon0)
        
        combined_x = yolo_weight * yolo_x + flow_weight * flow_x
        combined_y = yolo_weight * yolo_y + flow_weight * flow_y
        
        lat, lon = xy_m_to_latlon(combined_x, combined_y, self.lat0, self.lon0)
        return LatLon(lat, lon)
    
    def _calculate_yolo_confidence(self, class_weights: List[float], class_locs: List[LatLon]) -> float:
        """Calculate confidence for YOLO-based prediction"""
        if not class_weights:
            return 0.0
            
        # Base confidence from detection weights
        base_confidence = np.mean(class_weights)
        
        # Bonus for multiple detections
        detection_bonus = min(len(class_weights) / 3.0, 1.0) * 0.2
        
        # Penalty for large spread in detections
        if len(class_locs) > 1:
            distances = []
            for i in range(len(class_locs)):
                for j in range(i + 1, len(class_locs)):
                    dist = haversine_m(class_locs[i].lat, class_locs[i].lon,
                                     class_locs[j].lat, class_locs[j].lon)
                    distances.append(dist)
            if distances:
                avg_distance = np.mean(distances)
                spread_penalty = min(avg_distance / 200.0, 1.0) * 0.3
            else:
                spread_penalty = 0.0
        else:
            spread_penalty = 0.0
            
        confidence = base_confidence + detection_bonus - spread_penalty
        return max(0.0, min(1.0, confidence))
    
    def _is_prediction_valid(self, prediction: LatLon, confidence: float) -> bool:
        """Check if prediction is valid (not an outlier)"""
        if self.last_position is None or confidence < self.config.min_detection_confidence:
            return True  # Can't validate without reference
            
        # Check for large jumps
        distance = haversine_m(self.last_position.lat, self.last_position.lon,
                             prediction.lat, prediction.lon)
        
        if distance > self.max_jump_distance:
            return False
            
        # Check against velocity history
        if len(self.velocity_history) > 0:
            expected_displacement = np.mean([v[0]**2 + v[1]**2 for v in self.velocity_history]) * self.last_dt
            if distance > expected_displacement * 3.0:  # 3x expected movement
                return False
                
        return True
    
    def _update_history(self, position: LatLon, yolo_confidence: float, flow_confidence: float):
        """Update position and confidence history"""
        self.position_history.append(position)
        self.confidence_history.append((yolo_confidence, flow_confidence))
        
        # Update velocity
        if len(self.position_history) >= 2:
            prev_pos = self.position_history[-2]
            curr_pos = self.position_history[-1]
            
            if self.last_dt > 0:
                dx, dy = latlon_to_xy_m(curr_pos.lat, curr_pos.lon, prev_pos.lat, prev_pos.lon)
                velocity = (dx / self.last_dt, dy / self.last_dt)
                self.velocity_history.append(velocity)
                self.last_velocity = velocity
    
    def update_flow(self, frame: np.ndarray, dt: float, altitude: float, 
                   fov_x_deg: float, fov_y_deg: float):
        """Update optical flow detector with current frame"""
        self.last_dt = dt
        self.flow_detector.update(frame, dt, altitude, fov_x_deg, fov_y_deg)
    
    def get_current_mode(self) -> str:
        """Get current prediction mode"""
        return self.current_mode
    
    def get_mode_confidence(self) -> float:
        """Get confidence in current mode"""
        return self.mode_confidence
    
    def get_flow_velocity(self) -> Tuple[float, float]:
        """Get current optical flow velocity"""
        return self.flow_detector.get_velocity()
    
    def get_flow_speed_kmh(self) -> float:
        """Get current optical flow speed in km/h"""
        return self.flow_detector.get_speed_kmh()
    
    def get_flow_heading_deg(self) -> float:
        """Get current optical flow heading in degrees"""
        return self.flow_detector.get_heading_deg()
    
    def is_flow_stable(self) -> bool:
        """Check if optical flow is stable"""
        return self.flow_detector.is_flow_stable()
    
    def reset(self):
        """Reset the predictor state"""
        self.position_history.clear()
        self.velocity_history.clear()
        self.confidence_history.clear()
        self.last_position = None
        self.last_velocity = (0.0, 0.0)
        self.current_mode = "yolo"
        self.mode_confidence = 0.0
        self.flow_detector.reset()
