import cv2
import pandas as pd
from collections import defaultdict
from typing import List, Tuple
from config import AppCfg
from core_types import LatLon, BoxDet
from srt import load_srt_latlon
from tiles import CachedHttpTileProvider
from hud import HudRenderer
from detector.yolo_ultra import YoloUltranyxDetector
from stability import StabilityGate
from estimator import PositionEstimator, WeightedBarycenterPredictor, EmaPositionSmoother, DisplacementSpeedEstimator, KalmanCvSmoother
from dedup import deduplicate_by_class
from video_io import Cv2VideoWriter
from geo import haversine_m
import traceback

def overlay_bottom_right(frame, overlay_img, pad=12):
    H, W = frame.shape[:2]
    h, w = overlay_img.shape[:2]
    x1 = W - w - pad
    y1 = H - h - pad
    frame[y1:y1+h, x1:x1+w] = overlay_img
    return frame

def _get_screen_resolution():
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080

def _resize_for_screen(img, margin_w=80, margin_h=120):
    sw, sh = _get_screen_resolution()
    h, w = img.shape[:2]
    max_w = max(sw - margin_w, 1)
    max_h = max(sh - margin_h, 1)
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img

def main():
    cfg = AppCfg()

    # --- Load label->GPS (pure IO) ---
    df = pd.read_excel(cfg.data.excel_locations_path, header=None)
    df.columns = ["Label", "Latitude", "Longitude"]
    label2gps = {str(r.Label): LatLon(float(r.Latitude), float(r.Longitude)) for _, r in df.iterrows()}
    nodes_all: List[Tuple[str, LatLon]] = [(lbl, ll) for lbl, ll in label2gps.items()]
    print(f"[INFO] Loaded {len(label2gps)} label->GPS entries")

    # --- SRT (ground truth positions for accuracy only) ---
    srt = load_srt_latlon(cfg.data.srt_path)
    if not srt:
        raise RuntimeError(f"No SRT entries parsed from {cfg.data.srt_path}")
    print(f"[INFO] Loaded {len(srt)} SRT frames")
    lat0, lon0 = srt[0]["lat"], srt[0]["lon"]
    # derive dt/fps from SRT DiffTime when available to avoid double-reading video
    dt_ms_vals = [e.get("dt_ms") for e in srt[:120] if isinstance(e.get("dt_ms"), (int, float))]
    if dt_ms_vals:
        avg_dt_ms = sum(dt_ms_vals) / len(dt_ms_vals)
        fps = 1000.0 / max(avg_dt_ms, 1e-3)
    else:
        fps = 30.0
    dt = 1.0 / float(fps)

    # --- Detector ---
    det = YoloUltranyxDetector(cfg.model)

    # --- Tile/HUD ---
    tp = CachedHttpTileProvider(
        base_url=cfg.hud.tile_provider_url,
        cache_dir=cfg.hud.tile_cache_dir,
        user_agent=cfg.hud.user_agent,
        tile_size=cfg.hud.tile_size
    )
    hud = HudRenderer(tp, cfg.hud.tile_size, cfg.hud.hud_size, cfg.hud.hud_zoom)

    # --- Stability / Estimator ---
    gate = StabilityGate(
        history_len=cfg.stability.history_len,
        presence_gamma=cfg.stability.presence_gamma,
        conf_alpha=cfg.stability.conf_alpha,
        ema_beta=cfg.stability.ema_beta,
        lock_thresh=cfg.stability.lock_thresh,
        unlock_thresh=cfg.stability.unlock_thresh,
    )
    # --- Estimator (Strategy-based) ---
    predictor = WeightedBarycenterPredictor(lat0, lon0)
    if cfg.algo.smoother == "kalman":
        smoother = KalmanCvSmoother(lat0, lon0)
        # when using Kalman, speed can be derived from the filter's velocity later; keep displacement but smooth it
        speed_est = DisplacementSpeedEstimator(speed_alpha=cfg.smooth.speed_alpha)
        print("[INFO] Smoother: Kalman (CV)")
    else:
        smoother = EmaPositionSmoother(alpha=cfg.smooth.pos_alpha)
        speed_est = DisplacementSpeedEstimator(speed_alpha=cfg.smooth.speed_alpha)
        print("[INFO] Smoother: EMA")
    est = PositionEstimator(predictor, smoother, speed_est)

    # --- Video Output (initialized on first frame to avoid double-reading for size) ---
    wtr = None
    W = H = None  # determined from first frame
    print(f"[INFO] Target FPS (from SRT): {fps:.2f}")

    path_pred: List[LatLon] = []
    spd_gt_state = 0.0  # EMA for SRT speed (m/s)
    vis_spd_kmh = 0.0   # visualized algo speed (km/h) after slew + quantize
    vis_spd_gt_kmh = 0.0  # visualized SRT speed (km/h) after slew + quantize
    frame_idx = 0

    try:
        for frame_det in det.stream(cfg.video.input_path):
            frame = frame_det.frame_bgr
            names = frame_det.names

            # lazily initialize writer with first frame size
            if wtr is None:
                H, W = frame.shape[:2]
                wtr = Cv2VideoWriter(cfg.video.output_path, fps=fps, size=(W, H))
                print(f"[INFO] Video parameters: {W}x{H} @ {fps:.2f}fps")
            # per-frame visual smoothing params (shared)
            kmh_rate = cfg.smooth.speed_max_kmh_rate
            quantum = cfg.smooth.speed_quantum_kmh
            delta_max = kmh_rate * dt

            # --- Deduplicate per class (except "other") ---
            kept, class_conf = deduplicate_by_class(frame_det.boxes, names, "other")
            # filter out 'other' from stability, drawing, and estimation
            class_conf = {k: v for k, v in class_conf.items() if k != "other"}

            # --- Stability update & hysteresis ---
            presence_now = {names[b.cls_id]: 1 for b in kept if names[b.cls_id] != "other"}
            score = gate.update(presence_now, class_conf)

            # Keep only locked classes
            kept_locked: List[BoxDet] = [b for b in kept if names[b.cls_id] != "other" and gate.is_locked(names[b.cls_id])]

            # --- Draw Ultralytics style overlays (minimal, no blur tricks) ---
            # Avoid rescaling; draw directly on `frame`.
            for b in kept_locked:
                x1, y1, x2, y2 = b.xyxy.astype(int).tolist()
                cname = names[b.cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(frame, f"{cname} {b.conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            # --- Class center dots + label GPS ---
            class_weights = []
            class_locs = []
            for b in kept_locked:
                x1, y1, x2, y2 = b.xyxy
                cx, cy = int((x1 + x2)/2), int((y1 + y2)/2)
                cname = names[b.cls_id]
                cv2.circle(frame, (cx, cy), 6, (50, 200, 50), -1)

                ll = label2gps.get(cname)
                if ll:
                    cv2.putText(frame, f"{ll.lat:.6f}, {ll.lon:.6f}", (cx+8, cy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 200, 50), 2, cv2.LINE_AA)
                    # weights: score * (1+0.5*area_norm) * conf
                    w = (score.get(cname, 0.0)) * (1.0 + 0.5 * (((x2-x1)*(y2-y1)) / (W*H))) * max(b.conf, 1e-3)
                    class_weights.append(w)
                    class_locs.append(ll)

            # --- Estimate position & speed ---
            pred_s, spd_mps, spd_kmh = est.estimate(class_weights, class_locs, dt)
            if pred_s is not None:
                path_pred.append(pred_s)

            # --- Overlays: PRED, SPD, SRT, ERR ---
            y0 = 30
            if pred_s is not None:
                cv2.putText(frame, f"PRED {pred_s.lat:.6f}, {pred_s.lon:.6f}", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
                y0 += 30
                # Visual rate limiting + quantization for smoother perception
                target_kmh = float(spd_kmh)
                dv = max(min(target_kmh - vis_spd_kmh, delta_max), -delta_max)
                vis_spd_kmh = vis_spd_kmh + dv
                vis_spd_kmh_draw = round(vis_spd_kmh / quantum) * quantum if quantum > 0 else vis_spd_kmh
                cv2.putText(frame, f"SPD  {vis_spd_kmh_draw:5.1f} km/h", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                y0 += 30

            if frame_idx < len(srt) and pred_s is not None:
                gt_lat = srt[frame_idx]["lat"]; gt_lon = srt[frame_idx]["lon"]
                err_m = haversine_m(pred_s.lat, pred_s.lon, gt_lat, gt_lon)
                # compute ground-truth speed from SRT (m/s)
                if frame_idx > 0:
                    prev_gt_lat = srt[frame_idx-1]["lat"]; prev_gt_lon = srt[frame_idx-1]["lon"]
                    d_gt = haversine_m(prev_gt_lat, prev_gt_lon, gt_lat, gt_lon)
                    # Prefer per-frame dt_ms if available
                    if "dt_ms" in srt[frame_idx]:
                        dt_gt = max(srt[frame_idx]["dt_ms"], 1e-3) / 1000.0
                    else:
                        dt_gt = dt
                    spd_gt_mps = d_gt / dt_gt if dt_gt > 0 else 0.0
                    # Smooth SRT speed to reduce flicker
                    a = cfg.smooth.speed_alpha
                    spd_gt_state = (1 - a) * spd_gt_state + a * spd_gt_mps
                else:
                    spd_gt_mps = 0.0
                    spd_gt_state = (1 - cfg.smooth.speed_alpha) * spd_gt_state
                cv2.putText(frame, f"SRT  {gt_lat:.6f}, {gt_lon:.6f}", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
                y0 += 30
                cv2.putText(frame, f"ERR  {err_m:6.1f} m", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)
                y0 += 30
                # Slew + quantize SRT speed for visual stability
                target_gt_kmh = float(spd_gt_state * 3.6)
                dv_gt = max(min(target_gt_kmh - vis_spd_gt_kmh, delta_max), -delta_max)
                vis_spd_gt_kmh = vis_spd_gt_kmh + dv_gt
                vis_spd_gt_kmh_draw = round(vis_spd_gt_kmh / quantum) * quantum if quantum > 0 else vis_spd_gt_kmh
                cv2.putText(frame, f"SPD_GT {vis_spd_gt_kmh_draw:5.1f} km/h", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150,200,255), 2, cv2.LINE_AA)

            # --- Camera center reticle ---
            ch, cw = frame.shape[:2]
            ccx, ccy = cw // 2, ch // 2
            cv2.circle(frame, (ccx, ccy), 10, (255,255,255), 2)
            cv2.line(frame, (ccx-18, ccy), (ccx-4, ccy), (255,255,255), 2)
            cv2.line(frame, (ccx+4, ccy), (ccx+18, ccy), (255,255,255), 2)
            cv2.line(frame, (ccx, ccy-18), (ccx, ccy-4), (255,255,255), 2)
            cv2.line(frame, (ccx, ccy+4), (ccx, ccy+18), (255,255,255), 2)

            # --- HUD ---
            if pred_s is not None:
                # filter nodes by radius on the fly (no side effects)
                nearby = []
                for name, ll in nodes_all:
                    d = haversine_m(pred_s.lat, pred_s.lon, ll.lat, ll.lon)
                    if d <= cfg.hud.building_radius_m:
                        nearby.append((name, ll))
                hud_img = hud.render(pred_s, path_pred, nearby)
                frame = overlay_bottom_right(frame, hud_img, pad=12)

            # --- IO ---
            if wtr is not None:
                wtr.write(frame)
            if cfg.video.preview:
                disp = _resize_for_screen(frame)
                cv2.imshow("YOLO + HUD + Stability + Speed", disp)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    except Exception as e:
        print("[ERROR] Unhandled exception in main loop:", e)
        traceback.print_exc()
        raise
    finally:
        if wtr is not None:
            wtr.close()
        if cfg.video.preview:
            cv2.destroyAllWindows()
        print(f"[DONE] Saved: {cfg.video.output_path}")

if __name__ == "__main__":
    main()
