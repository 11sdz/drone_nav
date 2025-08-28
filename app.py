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
from estimator import PositionEstimator
from dedup import deduplicate_by_class
from video_io import Cv2VideoReader, Cv2VideoWriter
from geo import haversine_m

def overlay_bottom_right(frame, overlay_img, pad=12):
    H, W = frame.shape[:2]
    h, w = overlay_img.shape[:2]
    x1 = W - w - pad
    y1 = H - h - pad
    frame[y1:y1+h, x1:x1+w] = overlay_img
    return frame

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
    est = PositionEstimator(lat0, lon0, pos_alpha=cfg.smooth.pos_alpha)

    # --- Video IO ---
    rdr = Cv2VideoReader()
    W, H, fps = rdr.open(cfg.video.input_path)
    dt = 1.0 / float(fps)
    wtr = Cv2VideoWriter(cfg.video.output_path, fps=fps, size=(W, H))
    print(f"[INFO] Video parameters: {W}x{H} @ {fps:.2f}fps")

    path_pred: List[LatLon] = []
    frame_idx = 0

    try:
        for frame_det in det.stream(cfg.video.input_path):
            frame = frame_det.frame_bgr
            names = frame_det.names

            # --- Deduplicate per class (except "other") ---
            kept, class_conf = deduplicate_by_class(frame_det.boxes, names, "other")

            # --- Stability update & hysteresis ---
            presence_now = {names[b.cls_id]: 1 for b in kept}
            score = gate.update(presence_now, class_conf)

            # Keep only locked classes
            kept_locked: List[BoxDet] = [b for b in kept if gate.is_locked(names[b.cls_id])]

            # --- Draw Ultralytics style overlays (minimal, no blur tricks) ---
            # Avoid rescaling; draw directly on `frame`.
            for b in kept_locked:
                x1, y1, x2, y2 = b.xyxy.astype(int).tolist()
                cname = names[b.cls_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)
                cv2.putText(frame, f"{cname} {b.conf:.2f}", (x1, max(0, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

            # --- Class center dots + label GPS ---
            areas = []
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
                cv2.putText(frame, f"SPD  {spd_mps:5.1f} m/s  ({spd_kmh:5.1f} km/h)", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                y0 += 30

            if frame_idx < len(srt) and pred_s is not None:
                gt_lat = srt[frame_idx]["lat"]; gt_lon = srt[frame_idx]["lon"]
                err_m = haversine_m(pred_s.lat, pred_s.lon, gt_lat, gt_lon)
                cv2.putText(frame, f"SRT  {gt_lat:.6f}, {gt_lon:.6f}", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
                y0 += 30
                cv2.putText(frame, f"ERR  {err_m:6.1f} m", (20, y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2, cv2.LINE_AA)

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
            wtr.write(frame)
            if cfg.video.preview:
                cv2.imshow("YOLO + HUD + Stability + Speed", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

    finally:
        rdr.close()
        wtr.close()
        if cfg.video.preview:
            cv2.destroyAllWindows()
        print(f"[DONE] Saved: {cfg.video.output_path}")

if __name__ == "__main__":
    main()
