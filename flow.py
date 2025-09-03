import cv2
import numpy as np
from typing import Tuple, Optional, List
from collections import deque
from dataclasses import dataclass
from core_types import LatLon
from geo import offset_latlon_by_m, haversine_m

@dataclass
class FlowConfig:
    max_features: int = 500
    quality_level: float = 0.01
    min_distance: int = 10
    block_size: int = 3
    win_size: int = 15
    max_level: int = 2
    criteria: Tuple[int, int, float] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    confidence_threshold: float = 0.3
    min_features: int = 10
    history_length: int = 10
    velocity_alpha: float = 0.3
    acceleration_alpha: float = 0.2

class OpticalFlowDetector:
    """
    Optical flow detector with north-facing constraints and confidence tracking.
    Assumes drone always faces north, so:
    - Right movement = East (positive X)
    - Left movement = West (negative X) 
    - Forward movement = North (positive Y)
    - Backward movement = South (negative Y)
    """
    
    def __init__(self, config: FlowConfig):
        self.config = config
        self.prev_gray = None
        self.prev_features = None
        self.feature_params = {
            'maxCorners': config.max_features,
            'qualityLevel': config.quality_level,
            'minDistance': config.min_distance,
            'blockSize': config.block_size
        }
        self.lk_params = {
            'winSize': (config.win_size, config.win_size),
            'maxLevel': config.max_level,
            'criteria': config.criteria
        }
        
        # History and state tracking
        self.flow_history = deque(maxlen=config.history_length)
        self.velocity_east = 0.0
        self.velocity_north = 0.0
        self.acceleration_east = 0.0
        self.acceleration_north = 0.0
        self.last_velocity_east = 0.0
        self.last_velocity_north = 0.0
        self.confidence = 0.0
        self.is_stable = False
        
        # North-facing constraints
        self.north_constraint_enabled = True
        self.max_lateral_ratio = 0.3  # Max ratio of lateral vs forward movement
        
    def reset(self):
        """Reset the optical flow detector"""
        self.prev_gray = None
        self.prev_features = None
        self.flow_history.clear()
        self.velocity_east = 0.0
        self.velocity_north = 0.0
        self.acceleration_east = 0.0
        self.acceleration_north = 0.0
        self.last_velocity_east = 0.0
        self.last_velocity_north = 0.0
        self.confidence = 0.0
        self.is_stable = False
        
    def update(self, frame: np.ndarray, dt: float, altitude: float, 
               fov_x_deg: float, fov_y_deg: float) -> Tuple[float, float, float]:
        """
        Update optical flow with current frame.
        Returns: (east_displacement_m, north_displacement_m, confidence)
        """
        if dt <= 0:
            dt = 1e-3
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0, 0.0
            
        # Detect features if we don't have enough
        if self.prev_features is None or len(self.prev_features) < self.config.min_features:
            self.prev_features = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, **self.feature_params
            )
            if self.prev_features is not None:
                self.prev_features = self.prev_features.reshape(-1, 1, 2)
        
        if self.prev_features is None or len(self.prev_features) < self.config.min_features:
            self.prev_gray = gray
            return 0.0, 0.0, 0.0
            
        # Calculate optical flow
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_features, None, **self.lk_params
        )
        
        # Filter good points
        good_old = self.prev_features[status == 1]
        good_new = next_pts[status == 1]
        
        if len(good_old) < self.config.min_features:
            self.prev_gray = gray
            return 0.0, 0.0, 0.0
            
        # Calculate displacement in pixels
        displacement_px = good_new - good_old
        mean_displacement = np.mean(displacement_px, axis=0)
        
        # Convert pixel displacement to meters
        east_px, north_px = mean_displacement[0], mean_displacement[1]
        
        # Convert to meters using altitude and FOV
        east_m, north_m = self._pixel_to_meters(east_px, north_px, altitude, fov_x_deg, fov_y_deg, frame.shape)
        
        # Apply north-facing constraints
        if self.north_constraint_enabled:
            east_m, north_m = self._apply_north_constraints(east_m, north_m)
        
        # Calculate confidence based on feature quality and consistency
        confidence = self._calculate_confidence(displacement_px, good_old, good_new)
        
        # Update history and velocity estimates
        self._update_history_and_velocity(east_m, north_m, confidence, dt)
        
        # Update features for next frame
        self.prev_features = good_new.reshape(-1, 1, 2)
        self.prev_gray = gray
        
        return east_m, north_m, confidence
    
    def _pixel_to_meters(self, east_px: float, north_px: float, altitude: float, 
                        fov_x_deg: float, fov_y_deg: float, frame_shape: Tuple[int, int, int]) -> Tuple[float, float]:
        """Convert pixel displacement to meters using altitude and FOV"""
        h, w = frame_shape[:2]
        
        # Calculate ground coverage at current altitude
        fov_x_rad = np.radians(fov_x_deg)
        fov_y_rad = np.radians(fov_y_deg)
        
        # Ground coverage in meters
        ground_width = 2 * altitude * np.tan(fov_x_rad / 2)
        ground_height = 2 * altitude * np.tan(fov_y_rad / 2)
        
        # Meters per pixel
        m_per_px_x = ground_width / w
        m_per_px_y = ground_height / h
        
        # Convert pixel displacement to meters
        east_m = east_px * m_per_px_x
        north_m = north_px * m_per_px_y
        
        return east_m, north_m
    
    def _apply_north_constraints(self, east_m: float, north_m: float) -> Tuple[float, float]:
        """Apply north-facing constraints to limit unrealistic lateral movement"""
        # Calculate movement magnitude and direction
        total_movement = np.sqrt(east_m**2 + north_m**2)
        if total_movement < 1e-6:
            return east_m, north_m
            
        # Calculate ratio of lateral vs forward movement
        lateral_movement = abs(east_m)
        forward_movement = abs(north_m)
        
        if forward_movement > 1e-6:
            lateral_ratio = lateral_movement / forward_movement
            if lateral_ratio > self.max_lateral_ratio:
                # Scale down lateral movement
                scale_factor = self.max_lateral_ratio / lateral_ratio
                east_m *= scale_factor
                
        return east_m, north_m
    
    def _calculate_confidence(self, displacement_px: np.ndarray, 
                            _good_old: np.ndarray, _good_new: np.ndarray) -> float:
        """Calculate confidence based on feature quality and consistency"""
        if len(displacement_px) < self.config.min_features:
            return 0.0
            
        # Calculate displacement consistency
        displacement_std = np.std(displacement_px, axis=0)
        
        # Consistency score (lower std = higher confidence)
        consistency_score = 1.0 / (1.0 + np.mean(displacement_std))
        
        # Feature count score
        feature_score = min(len(displacement_px) / self.config.max_features, 1.0)
        
        # Movement magnitude score (too much or too little movement reduces confidence)
        movement_magnitude = np.sqrt(np.mean(displacement_px**2))
        if movement_magnitude < 0.5:
            movement_score = movement_magnitude / 0.5
        elif movement_magnitude > 10.0:
            movement_score = 10.0 / movement_magnitude
        else:
            movement_score = 1.0
            
        # Combined confidence
        confidence = consistency_score * feature_score * movement_score
        return min(confidence, 1.0)
    
    def _update_history_and_velocity(self, east_m: float, north_m: float, confidence: float, dt: float):
        """Update flow history and calculate velocity/acceleration"""
        # Store in history
        self.flow_history.append({
            'east_m': east_m,
            'north_m': north_m,
            'confidence': confidence,
            'dt': dt
        })
        
        # Update velocity estimates
        if dt > 0:
            current_vel_east = east_m / dt
            current_vel_north = north_m / dt
            
            # Smooth velocity with EMA
            alpha_vel = self.config.velocity_alpha
            self.velocity_east = (1 - alpha_vel) * self.velocity_east + alpha_vel * current_vel_east
            self.velocity_north = (1 - alpha_vel) * self.velocity_north + alpha_vel * current_vel_north
            
            # Calculate acceleration
            if abs(self.last_velocity_east) > 1e-6 or abs(self.last_velocity_north) > 1e-6:
                acc_east = (current_vel_east - self.last_velocity_east) / dt
                acc_north = (current_vel_north - self.last_velocity_north) / dt
                
                # Smooth acceleration with EMA
                alpha_acc = self.config.acceleration_alpha
                self.acceleration_east = (1 - alpha_acc) * self.acceleration_east + alpha_acc * acc_east
                self.acceleration_north = (1 - alpha_acc) * self.acceleration_north + alpha_acc * acc_north
            
            self.last_velocity_east = current_vel_east
            self.last_velocity_north = current_vel_north
        
        # Update overall confidence and stability
        self.confidence = confidence
        self.is_stable = self._check_stability()
    
    def _check_stability(self) -> bool:
        """Check if optical flow is stable based on history"""
        if len(self.flow_history) < 5:
            return False
            
        # Check if recent confidences are consistently high
        recent_confidences = [entry['confidence'] for entry in list(self.flow_history)[-5:]]
        avg_confidence = np.mean(recent_confidences)
        
        return avg_confidence >= self.config.confidence_threshold
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current velocity estimate (east, north) in m/s"""
        return self.velocity_east, self.velocity_north
    
    def get_acceleration(self) -> Tuple[float, float]:
        """Get current acceleration estimate (east, north) in m/s²"""
        return self.acceleration_east, self.acceleration_north
    
    def get_speed_kmh(self) -> float:
        """Get current speed in km/h"""
        speed_mps = np.sqrt(self.velocity_east**2 + self.velocity_north**2)
        return speed_mps * 3.6
    
    def get_heading_deg(self) -> float:
        """Get current heading in degrees (0° = North, 90° = East)"""
        if abs(self.velocity_east) < 1e-6 and abs(self.velocity_north) < 1e-6:
            return 0.0
        return np.degrees(np.arctan2(self.velocity_east, self.velocity_north))
    
    def predict_position(self, last_position: LatLon, dt: float) -> Optional[LatLon]:
        """Predict next position using current velocity and acceleration"""
        if last_position is None:
            return None
            
        # Predict displacement using velocity and acceleration
        pred_east = self.velocity_east * dt + 0.5 * self.acceleration_east * dt**2
        pred_north = self.velocity_north * dt + 0.5 * self.acceleration_north * dt**2
        
        # Apply to last position
        lat, lon = offset_latlon_by_m(last_position.lat, last_position.lon, pred_north, pred_east)
        return LatLon(lat, lon)
    
    def get_flow_confidence(self) -> float:
        """Get current flow confidence"""
        return self.confidence
    
    def is_flow_stable(self) -> bool:
        """Check if flow is currently stable"""
        return self.is_stable
