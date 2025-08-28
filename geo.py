import math

EARTH_R_M = 6_371_000.0

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlmb/2)**2
    return 2 * EARTH_R_M * math.asin(math.sqrt(a))

def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float):
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * m_per_deg_lon
    y = (lat - lat0) * m_per_deg_lat
    return x, y

def xy_m_to_latlon(x: float, y: float, lat0: float, lon0: float):
    m_per_deg_lat = 110_540.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    lat = lat0 + y / m_per_deg_lat
    lon = lon0 + x / m_per_deg_lon
    return lat, lon

# Web-mercator <-> global pixels
def latlon_to_global_px(lat: float, lon: float, z: int, tile_size: int):
    siny = math.sin(math.radians(lat))
    siny = min(max(siny, -0.9999), 0.9999)
    scale = tile_size * (2 ** z)
    x = (lon + 180.0) / 360.0 * scale
    y = (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi)) * scale
    return x, y

def global_px_to_tile_xy(px: float, py: float, tile_size: int):
    tx, ty = int(px // tile_size), int(py // tile_size)
    ox, oy = int(px % tile_size), int(py % tile_size)
    return tx, ty, ox, oy
