#!/usr/bin/env python3
"""
Oakland illegal dumping parser + map visualizer for:
https://data.oaklandca.gov/api/v3/views/dazd-43dc/query.json

Outputs:
- illegal_dumping_map.html (interactive Leaflet map)
- illegal_dumping_points.csv (parsed points)

Supports:
- Native lat/lon fields
- Geo fields like {"latitude": "...", "longitude": "..."}
- Web Mercator meters (EPSG:3857) "SRX/SRY" style fields (auto-detected)
- Unknown schema: inspects records and tries multiple coordinate strategies
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import folium
from folium.plugins import MarkerCluster, HeatMap

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None  # type: ignore


ENDPOINT = "https://data.oaklandca.gov/api/v3/views/dazd-43dc/query.json"


@dataclass
class Point:
    lat: float
    lon: float
    source: str
    props: Dict[str, Any]


def _is_floatish(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _looks_like_wgs84(lat: float, lon: float) -> bool:
    return -90 <= lat <= 90 and -180 <= lon <= 180


def _looks_like_web_mercator(x: float, y: float) -> bool:
    # EPSG:3857 meters, roughly:
    return abs(x) > 1000 and abs(y) > 1000 and abs(x) < 3e7 and abs(y) < 3e7


def _mercator_to_wgs84(x: float, y: float) -> Optional[Tuple[float, float]]:
    # Prefer pyproj if available.
    if Transformer is not None:
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x, y)
        if _looks_like_wgs84(lat, lon):
            return lat, lon

    # Fallback approximate conversion (works fine for Leaflet-level visualization)
    # Formula for spherical mercator:
    # lon = x / R * 180/pi
    # lat = (2*atan(exp(y/R)) - pi/2) * 180/pi
    import math

    R = 6378137.0
    lon = (x / R) * (180.0 / math.pi)
    lat = (2.0 * math.atan(math.exp(y / R)) - (math.pi / 2.0)) * (180.0 / math.pi)
    if _looks_like_wgs84(lat, lon):
        return lat, lon
    return None

def debug_dump(records: List[Dict[str, Any]], n: int = 3) -> None:
    import itertools
    import collections

    print("\n=== DEBUG: sample records (pretty) ===")
    for i, rec in enumerate(records[:n]):
        print(f"\n--- record[{i}] ---")
        try:
            print(json.dumps(rec, indent=2, sort_keys=True, default=str)[:4000])
        except Exception:
            print(str(rec)[:4000])

    # Flatten first record keys
    flat0 = _flatten(records[0]) if records else {}
    print("\n=== DEBUG: first record flattened keys (first 60) ===")
    for k in list(flat0.keys())[:60]:
        print(" ", k)

    # Search all records for coord-ish keys
    patterns = [
        "lat", "latitude", "lon", "lng", "long", "longitude",
        "geo", "geometry", "point", "location", "address",
        "srx", "sry", "x", "y", "easting", "northing"
    ]

    key_counter = collections.Counter()
    example_values = collections.defaultdict(list)

    for rec in records:
        flat = _flatten(rec)
        for k, v in flat.items():
            kl = k.lower()
            if any(p in kl for p in patterns):
                key_counter[k] += 1
                if len(example_values[k]) < 3 and v is not None:
                    example_values[k].append(v)

    print("\n=== DEBUG: coord-ish keys (top 50 by frequency) ===")
    for k, c in key_counter.most_common(50):
        print(f"{c:6d}  {k}   examples: {example_values.get(k)}")

    # Also show if there are any nested dicts that might hold geometry
    print("\n=== DEBUG: fields that are dict-like in first 50 records ===")
    dict_fields = collections.Counter()
    for rec in records[:50]:
        for k, v in rec.items():
            if isinstance(v, dict):
                dict_fields[k] += 1
    for k, c in dict_fields.most_common():
        print(f"{c:3d}  {k}")


def fetch_all_records(limit: int = 50000) -> List[Dict[str, Any]]:
    """
    Fetch records from Socrata v3 views query endpoint.

    Handles response shapes:
      - List[dict] (top-level array)
      - {"results": [...]} or {"data": [...]} etc.

    Uses limit/offset pagination when supported. If the endpoint ignores offset and
    keeps returning the same page, we detect and stop.
    """
    headers = {
        "User-Agent": "oakland-illegal-dumping-map/1.0 (requests)",
        "Accept": "application/json",
    }

    app_token = os.getenv("SOCRATA_APP_TOKEN") or os.getenv("SODATA_APP_TOKEN")
    if app_token:
        headers["X-App-Token"] = app_token

    def extract_batch(payload: Any) -> List[Dict[str, Any]]:
        # Case 1: top-level array
        if isinstance(payload, list):
            # sometimes list items are lists (Socrata legacy row arrays); keep as dict-ish
            return payload  # type: ignore

        # Case 2: top-level object
        if isinstance(payload, dict):
            for key in ("results", "data", "rows", "records"):
                v = payload.get(key)
                if isinstance(v, list):
                    return v  # type: ignore

        raise RuntimeError(
            "Unexpected response format: expected list or dict containing a list. "
            f"Got type={type(payload)}"
        )

    records: List[Dict[str, Any]] = []
    offset = 0
    page_size = min(5000, limit)

    # Track first record signature to detect non-paginating endpoints returning same page
    last_first_sig = None

    while True:
        params = {"limit": page_size, "offset": offset}

        r = requests.get(ENDPOINT, headers=headers, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()

        batch = extract_batch(payload)

        # If Socrata returns row-arrays, convert them to dicts if possible.
        # Otherwise keep them as-is; later flattening will still work for dicts.
        # (If you see list rows, tell me and I’ll add header-based mapping.)
        if len(batch) == 0:
            break

        # Detect "same page again" (offset ignored)
        first = batch[0]
        first_sig = json.dumps(first, sort_keys=True, default=str) if isinstance(first, (dict, list)) else str(first)
        if last_first_sig is not None and first_sig == last_first_sig:
            # offset is being ignored; avoid infinite loop
            # keep only what we have and stop
            break
        last_first_sig = first_sig

        # Truncate to limit
        remaining = limit - len(records)
        records.extend(batch[:remaining])

        if len(records) >= limit:
            records = records[:limit]
            break

        # If the server honored paging, this should move forward; otherwise we break above
        offset += len(batch)

        # If we got fewer than requested, likely end of data
        if len(batch) < page_size:
            break

    # Ensure each record is a dict for downstream code
    # If records are already dicts, fine; if they're lists, wrap them.
    normalized: List[Dict[str, Any]] = []
    for rec in records:
        if isinstance(rec, dict):
            normalized.append(rec)
        elif isinstance(rec, list):
            # preserve row array with indexed keys so downstream flattening works
            normalized.append({f"col_{i}": v for i, v in enumerate(rec)})
        else:
            normalized.append({"value": rec})

    return normalized


def _flatten(d: Any, prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dicts into a single dict with dotted keys.
    Lists are kept as-is.
    """
    out: Dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                out.update(_flatten(v, key))
            else:
                out[key] = v
    else:
        out[prefix or "value"] = d
    return out

def extract_points(records: List[Dict[str, Any]]) -> List[Point]:
    """
    Extract lat/lon from each record using multiple heuristics.
    Prioritize srx/sry if they look like WGS84 lon/lat.
    """
    points: List[Point] = []

    lat_keys = re.compile(r"(^|\.)(lat|latitude)$", re.IGNORECASE)
    lon_keys = re.compile(r"(^|\.)(lon|lng|long|longitude)$", re.IGNORECASE)

    # NOTE: we include srx/sry here but treat them specially (they may be lon/lat)
    x_keys = re.compile(r"(^|\.)(x|srx|mercatorx|easting)$", re.IGNORECASE)
    y_keys = re.compile(r"(^|\.)(y|sry|mercatory|northing)$", re.IGNORECASE)

    def pick_first_floatish(items: List[Tuple[str, Any]]) -> Optional[Tuple[str, float]]:
        for k, v in items:
            if _is_floatish(v):
                return k, float(v)
        return None

    for rec in records:
        flat = _flatten(rec)

        # 1) direct lat/lon fields
        lat_candidates = [(k, flat[k]) for k in flat.keys() if lat_keys.search(k)]
        lon_candidates = [(k, flat[k]) for k in flat.keys() if lon_keys.search(k)]
        lat_pick = pick_first_floatish(lat_candidates)
        lon_pick = pick_first_floatish(lon_candidates)

        if lat_pick and lon_pick:
            lat = lat_pick[1]
            lon = lon_pick[1]
            if _looks_like_wgs84(lat, lon):
                points.append(Point(
                    lat=lat,
                    lon=lon,
                    source="...",
                    props=dict(rec)   # <-- IMPORTANT: copy the record
                ))
                continue

        # 2) GeoJSON-ish coordinates list like reqaddress.coordinates = [lon, lat]
        # Flattening keeps lists intact.
        for k, v in flat.items():
            if isinstance(v, list) and len(v) >= 2 and _is_floatish(v[0]) and _is_floatish(v[1]):
                a = float(v[0])
                b = float(v[1])
                # GeoJSON order is [lon, lat]
                if _looks_like_wgs84(b, a):
                    points.append(Point(
                        lat=b,
                        lon=a,
                        source="...",
                        props=dict(rec)   # <-- IMPORTANT: copy the record
                    ))                    
                    break
                # Sometimes it's [lat, lon]
                if _looks_like_wgs84(a, b):
                    points.append(Point(
                        lat=a,
                        lon=b,
                        source="...",
                        props=dict(rec)   # <-- IMPORTANT: copy the record
                    ))                          
                    break
        else:
            # only runs if we didn't break (no coords parsed yet)
            pass
        if points and points[-1].props is rec:
            continue

        # 3) SRX/SRY or X/Y fields:
        #    - If they look like WGS84 lon/lat, use them.
        #    - Else if they look like Web Mercator meters, convert from EPSG:3857.
        x_candidates = [(k, flat[k]) for k in flat.keys() if x_keys.search(k)]
        y_candidates = [(k, flat[k]) for k in flat.keys() if y_keys.search(k)]
        x_pick = pick_first_floatish(x_candidates)
        y_pick = pick_first_floatish(y_candidates)

        if x_pick and y_pick:
            x = x_pick[1]
            y = y_pick[1]

            # Treat as lon/lat if plausible
            # (Your dataset: srx=-122..., sry=37...)
            if _looks_like_wgs84(y, x):
                points.append(Point(
                    lat=y,
                    lon=x,
                    source="...",
                    props=dict(rec)   # <-- IMPORTANT: copy the record
                ))                                
                continue

            # Otherwise treat as Web Mercator meters
            if _looks_like_web_mercator(x, y):
                latlon = _mercator_to_wgs84(x, y)
                if latlon:
                    lat, lon = latlon
                    points.append(Point(
                        lat=lat,
                        lon=lon,
                        source="...",
                        props=dict(rec)   # <-- IMPORTANT: copy the record
                    ))                         
                    continue

        # 4) String parsing fallback "POINT (... ...)" etc.
        for k, v in flat.items():
            if isinstance(v, str) and ("(" in v or "," in v):
                m = re.findall(r"-?\d+\.\d+|-?\d+", v)
                if len(m) >= 2:
                    a = float(m[0])
                    b = float(m[1])
                    # try (lon, lat)
                    if _looks_like_wgs84(b, a):
                        points.append(Point(
                            lat=b,
                            lon=a,
                            source="...",
                            props=dict(rec)   # <-- IMPORTANT: copy the record
                        ))                                
                        break
                    # try (lat, lon)
                    if _looks_like_wgs84(a, b):
                        points.append(Point(
                            lat=a,
                            lon=b,
                            source="...",
                            props=dict(rec)   # <-- IMPORTANT: copy the record
                        ))                               
                        break

    return points


def guess_popup_fields(df: pd.DataFrame) -> List[str]:
    """
    Try to pick useful columns for marker popups without knowing schema.
    """
    preferred = []
    for name in df.columns:
        n = name.lower()
        if any(s in n for s in ["address", "street", "location", "status", "type", "description", "request", "created", "date", "closed"]):
            preferred.append(name)
    # Keep it readable
    preferred = preferred[:8]
    return preferred


def make_map(points: List[Point], out_html: str = "illegal_dumping_map.html") -> None:
    if not points:
        raise RuntimeError("No points extracted. You may need to inspect schema/fields and adjust heuristics.")

    df = pd.DataFrame(
        [{"lat": p.lat, "lon": p.lon, "coord_source": p.source, **_flatten(p.props)} for p in points]
    )

    # Basic cleanup: drop rows with missing coords
    df = df.dropna(subset=["lat", "lon"]).copy()

    # Ensure numeric
    df["lat"] = df["lat"].astype(float)
    df["lon"] = df["lon"].astype(float)

    # Sanity check: auto-fix swapped lat/lon
    if df["lat"].abs().mean() > 90 and df["lon"].abs().mean() <= 90:
        print("⚠️ Detected swapped lat/lon — fixing")
        df[["lat", "lon"]] = df[["lon", "lat"]]

    # Initialize map without forcing zoom
    min_lat = df["lat"].min()
    max_lat = df["lat"].max()
    min_lon = df["lon"].min()
    max_lon = df["lon"].max()

    print("Map bounds:", min_lat, min_lon, max_lat, max_lon)

    m = folium.Map(tiles="OpenStreetMap", prefer_canvas=True)

    m.fit_bounds([
        [min_lat, min_lon],
        [max_lat, max_lon],
    ], padding=(40, 40))


    # Marker clusters
    cluster = MarkerCluster(
        name="Sites (clustered)",
        disableClusteringAtZoom=15,
        maxClusterRadius=50,
    )
    popup_fields = guess_popup_fields(df)

    for _, row in df.iterrows():
        popup_parts = [f"<b>coord_source</b>: {row.get('coord_source','') }"]
        for col in popup_fields:
            val = row.get(col)

            if val is None:
                continue

            # Skip NaN (but allow lists/dicts)
            try:
                if pd.isna(val):
                    continue
            except Exception:
                pass

            # Render lists/dicts cleanly
            if isinstance(val, (list, dict)):
                val_str = json.dumps(val)
            else:
                val_str = str(val)

            popup_parts.append(f"<b>{col}</b>: {val_str[:200]}")

        popup_html = "<br/>".join(popup_parts)

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            weight=1,
            fill=True,
            fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=400),
        ).add_to(cluster)

    cluster.add_to(m)

    # Heatmap
    heat_data = df[["lat", "lon"]].values.tolist()
    HeatMap(heat_data, name="Heatmap", radius=10, blur=12).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # Save artifacts
    df.to_csv("illegal_dumping_points.csv", index=False)
    m.save(out_html)

    print(f"Wrote: {out_html}")
    print("Wrote: illegal_dumping_points.csv")
    print(f"Extracted points: {len(df):,}")
    print("Tip: open the HTML file in your browser.")


def main():
    limit = int(os.getenv("LIMIT", "50000"))
    print(f"Fetching records (limit={limit}) from:\n  {ENDPOINT}")
    records = fetch_all_records(limit=limit)
    print(f"Fetched records: {len(records):,}")

    debug_dump(records, n=2)   # <-- add this line

    print("Extracting coordinates...")
    points = extract_points(records)
    print(f"Extracted mappable points: {len(points):,}")

    make_map(points)


if __name__ == "__main__":
    main()
