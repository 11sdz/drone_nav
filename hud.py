from typing import List, Tuple
import numpy as np
import cv2
from core_types import IHUDRenderer, LatLon
from geo import latlon_to_global_px, global_px_to_tile_xy

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

    def render(self, center: LatLon, path: List[LatLon], nodes: List[Tuple[str, LatLon]]) -> np.ndarray:
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

        return hud
