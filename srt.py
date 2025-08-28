import re
from typing import List, Dict

SRT_LAT_RE = re.compile(r"\[latitude:\s*([-\d\.]+)\]")
SRT_LON_RE = re.compile(r"\[longitude:\s*([-\d\.]+)\]")

def load_srt_latlon(path: str) -> List[Dict[str, float]]:
    entries, block = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if block:
                    text = " ".join(block)
                    la = SRT_LAT_RE.search(text); lo = SRT_LON_RE.search(text)
                    if la and lo:
                        entries.append({"lat": float(la.group(1)), "lon": float(lo.group(1))})
                    block = []
            else:
                block.append(line.strip())
        if block:
            text = " ".join(block)
            la = SRT_LAT_RE.search(text); lo = SRT_LON_RE.search(text)
            if la and lo:
                entries.append({"lat": float(la.group(1)), "lon": float(lo.group(1))})
    return entries
