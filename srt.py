import re
from typing import List, Dict

SRT_LAT_RE = re.compile(r"\[latitude:\s*([-\d\.]+)\]")
SRT_LON_RE = re.compile(r"\[longitude:\s*([-\d\.]+)\]")
SRT_DIFF_RE = re.compile(r"DiffTime\s*:\s*(\d+)ms")
SRT_ALT_RE = re.compile(r"\[altitude:\s*([-\d\.]+)\]")

def load_srt_latlon(path: str) -> List[Dict[str, float]]:
    entries, block = [], []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() == "":
                if block:
                    text = " ".join(block)
                    la = SRT_LAT_RE.search(text); lo = SRT_LON_RE.search(text)
                    dt = SRT_DIFF_RE.search(text)
                    alt = SRT_ALT_RE.search(text)
                    if la and lo:
                        entry: Dict[str, float] = {"lat": float(la.group(1)), "lon": float(lo.group(1))}
                        if dt:
                            try:
                                entry["dt_ms"] = float(dt.group(1))
                            except Exception:
                                pass
                        if alt:
                            try:
                                entry["alt"] = float(alt.group(1))
                            except Exception:
                                pass
                        entries.append(entry)
                    block = []
            else:
                block.append(line.strip())
        if block:
            text = " ".join(block)
            la = SRT_LAT_RE.search(text); lo = SRT_LON_RE.search(text)
            dt = SRT_DIFF_RE.search(text)
            alt = SRT_ALT_RE.search(text)
            if la and lo:
                entry2: Dict[str, float] = {"lat": float(la.group(1)), "lon": float(lo.group(1))}
                if dt:
                    try:
                        entry2["dt_ms"] = float(dt.group(1))
                    except Exception:
                        pass
                if alt:
                    try:
                        entry2["alt"] = float(alt.group(1))
                    except Exception:
                        pass
                entries.append(entry2)
    return entries
