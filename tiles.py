import os
import cv2
import numpy as np
import requests
from typing import Optional
from core_types import ITileProvider

class CachedHttpTileProvider(ITileProvider):
    def __init__(self, base_url: str, cache_dir: str, user_agent: str, tile_size: int):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.tile_size = tile_size
        self.headers = {"User-Agent": user_agent}
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, z: int, x: int, y: int) -> str:
        return os.path.join(self.cache_dir, f"{z}_{x}_{y}.png")

    def get_tile(self, z: int, x: int, y: int) -> Optional[np.ndarray]:
        path = self._cache_path(z, x, y)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        url = self.base_url.format(z=z, x=x, y=y)
        try:
            r = requests.get(url, headers=self.headers, timeout=5)
            r.raise_for_status()
            data = np.frombuffer(r.content, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                cv2.imwrite(path, img)
            return img
        except Exception:
            return None
