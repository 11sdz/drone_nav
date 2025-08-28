from typing import List, Tuple
import numpy as np
import cv2
from core_types import IHUDRenderer, LatLon
from geo import latlon_to_global_px, global_px_to_tile_xy, haversine_m, latlon_to_xy_m

class HudRenderer(IHUDRenderer):
    def __init__(self, tile_provider, tile_size: int, hud_size: int, zoom: int):
        self.tp = tile_provider
        self.tile_size = tile_size
        self.hud_size = hud_size
        self.zoom = zoom

    def _build_canvas(self, center: LatLon):
        cx, cy = latlon_to_global_px(center.lat, center.lon, self.zoom, self.tile_size)
        half = self.hud_size // 2
        tl_px = (cx - half, cy - half)
        br_px = (cx + half, cy + half)

        tl_tx, tl_ty, _, _ = global_px_to_tile_xy(*tl_px, self.tile_size)
        br_tx, br_ty, _, _ = global_px_to_tile_xy(*br_px, self.tile_size)

        canvas = np.zeros((self.hud_size, self.hud_size, 3), dtype=np.uint8)
        for ty in range(tl_ty, br_ty + 1):
            for tx in range(tl_tx, br_tx + 1):
                tile = self.tp.get_tile(self.zoom, tx, ty)
                if tile is None:
                    continue
                tile_px_x, tile_px_y = tx * self.tile_size, ty * self.tile_size
                x0 = max(tile_px_x, tl_px[0]); y0 = max(tile_px_y, tl_px[1])
                x1 = min(tile_px_x + self.tile_size, br_px[0]); y1 = min(tile_px_y + self.tile_size, br_px[1])
                if x1 <= x0 or y1 <= y0:
                    continue
                sx0 = int(x0 - tile_px_x); sy0 = int(y0 - tile_px_y)
                sx1 = int(x1 - tile_px_x); sy1 = int(y1 - tile_px_y)
                crop = tile[sy0:sy1, sx0:sx1]
                dx0 = int(x0 - tl_px[0]); dy0 = int(y0 - tl_px[1])
                dx1 = dx0 + crop.shape[1]; dy1 = dy0 + crop.shape[0]
                canvas[dy0:dy1, dx0:dx1] = crop

        cv2.rectangle(canvas, (0,0), (self.hud_size-1,self.hud_size-1), (220,220,220), 1)
        cv2.arrowedLine(canvas, (30, 60), (30, 20), (255,255,255), 2, tipLength=0.35)
        cv2.putText(canvas, "N", (22, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        return canvas, (cx, cy)

    def _project(self, p: LatLon, center_global_px):
        px, py = latlon_to_global_px(p.lat, p.lon, self.zoom, self.tile_size)
        cx, cy = center_global_px
        u = int(self.hud_size // 2 + (px - cx))
        v = int(self.hud_size // 2 + (py - cy))
        return u, v

    def render(self, center: LatLon, path: List[LatLon], nodes: List[Tuple[str, LatLon]], actual: LatLon = None) -> np.ndarray:
        hud, center_global_px = self._build_canvas(center)

        # path polyline
        if len(path) >= 2:
            pts = [self._project(p, center_global_px) for p in path]
            for i in range(1, len(pts)):
                cv2.line(hud, pts[i-1], pts[i], (0,255,0), 2)
            cv2.circle(hud, pts[0], 4, (0,0,255), -1)
            cv2.circle(hud, pts[-1], 5, (0,255,0), -1)

        # building nodes
        for name, ll in nodes:
            u, v = self._project(ll, center_global_px)
            if 0 <= u < self.hud_size and 0 <= v < self.hud_size:
                cv2.circle(hud, (u, v), 4, (255,255,255), -1)
                cv2.putText(hud, name, (u+5, v-5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

        # camera vs barycenter triangle/vector
        if len(path) >= 1:
            up, vp = self._project(path[-1], center_global_px)  # barycenter-based predicted ground point
            uc, vc = self._project(center, center_global_px)    # camera-estimated position (HUD center)
            # hypotenuse (magenta)
            cv2.line(hud, (up, vp), (uc, vc), (255, 0, 255), 2)
            # legs (thin magenta)
            cv2.line(hud, (up, vp), (uc, vp), (255, 0, 255), 1)
            cv2.line(hud, (uc, vp), (uc, vc), (255, 0, 255), 1)
            # labels: components and total distance in meters
            dx_m, dy_m = latlon_to_xy_m(path[-1].lat, path[-1].lon, center.lat, center.lon)
            err_hyp = int(round((dx_m**2 + dy_m**2) ** 0.5))
            midx, midy = (up + uc) // 2, (vp + vc) // 2
            cv2.putText(hud, f"dE {dx_m:+.0f} m", (min(up, uc) + 4, vp - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"dN {dy_m:+.0f} m", (uc + 4, min(vp, vc) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,0,255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"|d| {err_hyp} m", (midx + 6, midy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1, cv2.LINE_AA)
            # camera marker
            cv2.circle(hud, (uc, vc), 5, (255, 0, 255), -1)
            cv2.putText(hud, "CAM", (uc + 6, vc - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,0,255), 1, cv2.LINE_AA)

        # actual drone location marker (from SRT) + error vector
        if actual is not None:
            ua, va = self._project(actual, center_global_px)
            if 0 <= ua < self.hud_size and 0 <= va < self.hud_size:
                # predicted position on HUD: last path point if available, else center
                if len(path) >= 1:
                    up, vp = self._project(path[-1], center_global_px)
                else:
                    up, vp = self.hud_size // 2, self.hud_size // 2
                # error vector line
                cv2.line(hud, (up, vp), (ua, va), (0, 200, 255), 2)
                # actual marker
                cv2.circle(hud, (ua, va), 5, (0, 200, 255), -1)
                cv2.circle(hud, (ua, va), 9, (0, 200, 255), 2)
                cv2.putText(hud, "DRONE", (ua+6, va-6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1, cv2.LINE_AA)
                # numeric distance in meters
                err_m = haversine_m(center.lat, center.lon, actual.lat, actual.lon)
                cv2.putText(hud, f"ERR {err_m:4.0f} m", (8, self.hud_size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2, cv2.LINE_AA)

        return hud
