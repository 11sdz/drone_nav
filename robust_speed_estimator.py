from typing import Tuple, Optional
import numpy as np
from collections import deque
from dataclasses import dataclass
from core_types import LatLon, ISpeedEstimator
from geo import haversine_m
from flow import OpticalFlowDetector

@dataclass
class RobustSpeedConfig:
    # Position-based speed settings
    position_alpha: float = 0.2
    min_position_distance_m: float = 1.0
    
    # Flow-based speed settings
    flow_alpha: float = 0.3
    flow_weight: float = 0.7  # Weight of flow vs position
    
    # Speed filtering
    max_speed_kmh: float = 100.0
    min_speed_kmh: float = 0.1
    speed_smoothing_alpha: float = 0.15
    
    # History and validation
    history_length: int = 10
    outlier_rejection_enabled: bool = True
    max_speed_change_ratio: float = 0.5  # Max change per frame

class RobustSpeedEstimator(ISpeedEstimator):
    """
    Robust speed estimator that combines position-based and flow-based speed estimates
    with noise filtering and outlier rejection.
    """
    
    def __init__(self, config: RobustSpeedConfig, flow_detector: Optional[OpticalFlowDetector] = None):
        self.config = config
        self.flow_detector = flow_detector
        
        # Position-based speed tracking
        self.prev_position: Optional[LatLon] = None
        self.position_speed_state: float = 0.0
        self.position_speed_history = deque(maxlen=config.history_length)
        
        # Flow-based speed tracking
        self.flow_speed_state: float = 0.0
        self.flow_speed_history = deque(maxlen=config.history_length)
        
        # Combined speed tracking
        self.combined_speed_state: float = 0.0
        self.combined_speed_history = deque(maxlen=config.history_length)
        
        # State tracking
        self.last_dt: float = 1.0
        self.speed_confidence: float = 0.0
        
    def update(self, position: Optional[LatLon], dt: float) -> Tuple[float, float]:
        """
        Update speed estimate using position and/or optical flow.
        Returns: (speed_mps, speed_kmh)
        """
        if dt <= 0:
            dt = 1e-3
        self.last_dt = dt
        
        # Get position-based speed
        position_speed_mps = self._update_position_speed(position, dt)
        
        # Get flow-based speed
        flow_speed_mps = self._update_flow_speed()
        
        # Combine speeds
        combined_speed_mps = self._combine_speeds(position_speed_mps, flow_speed_mps)
        
        # Apply filtering and validation
        filtered_speed_mps = self._filter_speed(combined_speed_mps)
        
        # Update state
        self._update_speed_state(filtered_speed_mps)
        
        # Convert to km/h
        speed_kmh = filtered_speed_mps * 3.6
        
        return filtered_speed_mps, speed_kmh
    
    def _update_position_speed(self, position: Optional[LatLon], dt: float) -> float:
        """Update position-based speed estimate"""
        if position is None or self.prev_position is None:
            self.prev_position = position
            return self.position_speed_state
            
        # Calculate distance and speed
        distance_m = haversine_m(self.prev_position.lat, self.prev_position.lon,
                               position.lat, position.lon)
        
        if distance_m < self.config.min_position_distance_m:
            # Too small movement, use previous speed
            return self.position_speed_state
            
        speed_mps = distance_m / dt
        
        # Apply EMA smoothing
        if self.config.position_alpha > 0:
            self.position_speed_state = (1 - self.config.position_alpha) * self.position_speed_state + \
                                      self.config.position_alpha * speed_mps
        else:
            self.position_speed_state = speed_mps
            
        # Store in history
        self.position_speed_history.append(speed_mps)
        
        self.prev_position = position
        return self.position_speed_state
    
    def _update_flow_speed(self) -> float:
        """Update flow-based speed estimate"""
        if self.flow_detector is None:
            return 0.0
            
        flow_speed_mps = self.flow_detector.get_speed_kmh() / 3.6
        
        # Apply EMA smoothing
        if self.config.flow_alpha > 0:
            self.flow_speed_state = (1 - self.config.flow_alpha) * self.flow_speed_state + \
                                  self.config.flow_alpha * flow_speed_mps
        else:
            self.flow_speed_state = flow_speed_mps
            
        # Store in history
        self.flow_speed_history.append(flow_speed_mps)
        
        return self.flow_speed_state
    
    def _combine_speeds(self, position_speed_mps: float, flow_speed_mps: float) -> float:
        """Combine position-based and flow-based speed estimates"""
        
        # Determine weights based on availability and confidence
        if flow_speed_mps > 0 and self.flow_detector is not None:
            flow_confidence = self.flow_detector.get_flow_confidence()
            flow_stable = self.flow_detector.is_flow_stable()
            
            if flow_stable and flow_confidence > 0.3:
                # Flow is reliable, use weighted combination
                flow_weight = self.config.flow_weight * flow_confidence
                position_weight = 1.0 - flow_weight
            else:
                # Flow not reliable, use position
                flow_weight = 0.0
                position_weight = 1.0
        else:
            # No flow available, use position
            flow_weight = 0.0
            position_weight = 1.0
            
        # Combine speeds
        if position_speed_mps > 0 and flow_speed_mps > 0:
            combined_speed = position_weight * position_speed_mps + flow_weight * flow_speed_mps
        elif position_speed_mps > 0:
            combined_speed = position_speed_mps
        elif flow_speed_mps > 0:
            combined_speed = flow_speed_mps
        else:
            combined_speed = 0.0
            
        return combined_speed
    
    def _filter_speed(self, speed_mps: float) -> float:
        """Apply filtering and validation to speed estimate"""
        
        # Convert to km/h for filtering
        speed_kmh = speed_mps * 3.6
        
        # Apply speed limits
        speed_kmh = max(self.config.min_speed_kmh, min(speed_kmh, self.config.max_speed_kmh))
        
        # Outlier rejection
        if self.config.outlier_rejection_enabled and len(self.combined_speed_history) > 0:
            speed_kmh = self._reject_outliers(speed_kmh)
            
        # Convert back to m/s
        return speed_kmh / 3.6
    
    def _reject_outliers(self, speed_kmh: float) -> float:
        """Reject speed outliers based on history"""
        if len(self.combined_speed_history) < 3:
            return speed_kmh
            
        # Calculate expected speed range based on history
        recent_speeds = list(self.combined_speed_history)[-5:]
        mean_speed = np.mean(recent_speeds) * 3.6  # Convert to km/h
        std_speed = np.std(recent_speeds) * 3.6
        
        # Check for large changes
        if len(self.combined_speed_history) > 0:
            last_speed = self.combined_speed_history[-1] * 3.6
            max_change = last_speed * self.config.max_speed_change_ratio
            
            if abs(speed_kmh - last_speed) > max_change:
                # Reject large changes, use previous speed
                return last_speed
                
        # Check for statistical outliers
        if std_speed > 0:
            z_score = abs(speed_kmh - mean_speed) / std_speed
            if z_score > 2.0:  # 2-sigma outlier
                # Use mean instead of outlier
                return mean_speed
                
        return speed_kmh
    
    def _update_speed_state(self, speed_mps: float):
        """Update the combined speed state"""
        # Apply final smoothing
        if self.config.speed_smoothing_alpha > 0:
            self.combined_speed_state = (1 - self.config.speed_smoothing_alpha) * self.combined_speed_state + \
                                      self.config.speed_smoothing_alpha * speed_mps
        else:
            self.combined_speed_state = speed_mps
            
        # Store in history
        self.combined_speed_history.append(speed_mps)
        
        # Update confidence based on consistency
        self._update_confidence()
    
    def _update_confidence(self):
        """Update speed confidence based on consistency"""
        if len(self.combined_speed_history) < 3:
            self.speed_confidence = 0.5
            return
            
        # Calculate consistency
        recent_speeds = list(self.combined_speed_history)[-5:]
        std_speed = np.std(recent_speeds)
        mean_speed = np.mean(recent_speeds)
        
        if mean_speed > 0:
            cv = std_speed / mean_speed  # Coefficient of variation
            consistency = 1.0 / (1.0 + cv)  # Higher consistency = lower CV
        else:
            consistency = 0.0
            
        # Factor in flow stability if available
        if self.flow_detector is not None:
            flow_stable = self.flow_detector.is_flow_stable()
            flow_confidence = self.flow_detector.get_flow_confidence()
            flow_factor = 0.5 + 0.5 * (flow_confidence if flow_stable else 0.0)
        else:
            flow_factor = 0.5
            
        self.speed_confidence = consistency * flow_factor
    
    def get_speed_confidence(self) -> float:
        """Get confidence in current speed estimate"""
        return self.speed_confidence
    
    def get_position_speed_kmh(self) -> float:
        """Get position-based speed in km/h"""
        return self.position_speed_state * 3.6
    
    def get_flow_speed_kmh(self) -> float:
        """Get flow-based speed in km/h"""
        return self.flow_speed_state * 3.6
    
    def get_combined_speed_kmh(self) -> float:
        """Get combined speed in km/h"""
        return self.combined_speed_state * 3.6
    
    def reset(self):
        """Reset the speed estimator"""
        self.prev_position = None
        self.position_speed_state = 0.0
        self.flow_speed_state = 0.0
        self.combined_speed_state = 0.0
        self.speed_confidence = 0.0
        
        self.position_speed_history.clear()
        self.flow_speed_history.clear()
        self.combined_speed_history.clear()
