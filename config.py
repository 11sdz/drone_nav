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
    lock_thresh: float = 0.70
    unlock_thresh: float = 0.55

@dataclass(frozen=True)
class SmoothCfg:
    pos_alpha: float = 0.25  # temporal EMA

@dataclass(frozen=True)
class AppCfg:
    model: ModelCfg = ModelCfg()
    video: VideoCfg = VideoCfg()
    data: DataCfg = DataCfg()
    hud: HudCfg = HudCfg()
    stability: StabilityCfg = StabilityCfg()
    smooth: SmoothCfg = SmoothCfg()
