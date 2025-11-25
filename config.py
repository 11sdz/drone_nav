from dataclasses import dataclass

@dataclass(frozen=True)
class ModelCfg:
    weights_path: str = "data/best.pt"
    conf: float = 0.25
    img_size: int = 1280
    use_half: bool = True
    device_index: int = 0

@dataclass(frozen=True)
class VideoCfg:
    input_path: str = "inputs/DJI_0072wrongangle.MP4"
    output_path: str = "predictions/DJI_0072wrongangle_output.mp4"
    preview: bool = True

@dataclass(frozen=True)
class DataCfg:
    srt_path: str = "inputs/DJI_0072wrongangle.srt"
    excel_locations_path: str = "data/locations _ariel_uni.xlsx"  # Label | Latitude | Longitude

@dataclass(frozen=True)
class HudCfg:
    tile_size: int = 256
    hud_size: int = 320
    hud_zoom: int = 18
    building_radius_m: float = 400.0
    tile_provider_url: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    tile_cache_dir: str = "tile_cache"
    user_agent: str = "DroneHUD/1.0 (educational)"

@dataclass(frozen=True)
class StabilityCfg:
    history_len: int = 10
    presence_gamma: float = 1.5
    conf_alpha: float = 0.7
    ema_beta: float = 0.3
    lock_thresh: float = 0.65
    unlock_thresh: float = 0.6

@dataclass(frozen=True)
class SmoothCfg:
    pos_alpha: float = 0.25  # temporal EMA
    speed_alpha: float = 0.15  # EMA for speed smoothing (lower = smoother)
    speed_max_kmh_rate: float = 4.0  # max visual speed change rate (km/h per second)
    speed_quantum_kmh: float = 0.5   # round displayed speed to this step (km/h)

@dataclass(frozen=True)
class AlgoCfg:
    predictor: str = "robust"  # "weighted_barycenter", "robust"
    smoother: str = "kalman"  # "ema", "kalman"
    speed: str = "robust"  # "displacement", "robust"

@dataclass(frozen=True)
class CameraCfg:
    pitch_deg: float = 50.0  # gimbal down angle from horizontal (40-60 typical)
    heading_deg: float = 0.0 # north-locked
    fov_x_deg: float = 70.0  # horizontal field of view
    fov_y_deg: float = 45.0  # vertical field of view

@dataclass(frozen=True)
class FlowCfg:
    enabled: bool = True
    max_features: int = 500
    quality_level: float = 0.01
    min_distance: int = 10
    confidence_threshold: float = 0.4
    stability_threshold: float = 0.5
    history_length: int = 10
    velocity_alpha: float = 0.3
    acceleration_alpha: float = 0.2
    north_constraint_enabled: bool = True
    max_lateral_ratio: float = 0.3

@dataclass(frozen=True)
class RobustCfg:
    # YOLO settings
    yolo_confidence_threshold: float = 0.6
    yolo_stability_threshold: float = 0.7
    min_yolo_detections: int = 1
    
    # Flow settings
    flow_confidence_threshold: float = 0.4
    flow_stability_threshold: float = 0.5
    flow_primary_weight: float = 0.8
    
    # Noise filtering
    max_jump_distance_m: float = 100.0
    min_detection_confidence: float = 0.3
    outlier_rejection_enabled: bool = True
    
    # History and smoothing
    history_length: int = 10
    position_alpha: float = 0.3
    velocity_alpha: float = 0.2
    
    # Switching logic
    flow_fallback_enabled: bool = True
    hybrid_mode_enabled: bool = True

@dataclass(frozen=True)
class AppCfg:
    model: ModelCfg = ModelCfg()
    video: VideoCfg = VideoCfg()
    data: DataCfg = DataCfg()
    hud: HudCfg = HudCfg()
    stability: StabilityCfg = StabilityCfg()
    smooth: SmoothCfg = SmoothCfg()
    algo: AlgoCfg = AlgoCfg()
    camera: CameraCfg = CameraCfg()
    flow: FlowCfg = FlowCfg()
    robust: RobustCfg = RobustCfg()
    export_results: bool = True
    results_dir: str = "output/results"
