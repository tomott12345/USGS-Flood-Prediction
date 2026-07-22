"""Upstream gage discovery via USGS's Network Linked Data Index (NLDI).

Uses navigation mode "UT" (Upstream with Tributaries), which is the key
detail: it correctly crosses tributary boundaries. For a site sitting at a
confluence (like Pompton, formed by the Ramapo + Pequannock + Wanaque
Rivers), UT returns gages on *every* contributing tributary, not just the
mainstem -- unlike "UM" (upstream mainstem only), which would miss all but
one branch.

Most NLDI results are small monitoring sites with no live instantaneous-
values (IV) data (historical/periodic sampling only), so results are
filtered down to sites that actually have fetchable recent gage-height or
discharge data.
"""
import json
import os
import urllib.request
from datetime import datetime, timedelta

from data import fetch_historical_series

NLDI_BASE = "https://api.water.usgs.gov/nldi/linked-data/nwissite"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "upstream_cache")


def _nldi_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "USGS-Flood-Prediction"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())


def find_upstream_candidates(site_code, distance_km=50):
    """All gaged sites upstream (any tributary) within distance_km along the
    flow network (river-network distance, not straight-line). NLDI's result
    order turned out NOT to be reliably nearest-first in practice -- some
    genuinely close confluence gages showed up past index 150 of ~194 -- so
    find_upstream_gages below scans deep into the list rather than assuming
    the first handful of candidates are the closest ones.
    """
    url = f"{NLDI_BASE}/USGS-{site_code}/navigation/UT/nwissite?f=json&distance={distance_km}"
    data = _nldi_get(url)
    candidates = []
    for feature in data.get("features", []):
        props = feature["properties"]
        identifier = props.get("identifier", "")
        if not identifier.startswith("USGS-"):
            continue
        candidate_code = identifier.replace("USGS-", "")
        if candidate_code == site_code:
            continue  # NLDI's navigation graph includes the origin site itself
        candidates.append({"site_code": candidate_code, "name": props.get("name")})
    return candidates


def find_upstream_gages(site_code, distance_km=50, max_sites=5, check_days=3, max_candidates_checked=200, use_cache=True):
    """Walk NLDI candidates and keep the first max_sites that actually have
    live gage-height or discharge data. This involves one HTTP round-trip per
    candidate checked (~90s for Pompton's ~194 candidates), which the
    watershed's real network topology doesn't change often -- so the result
    is cached to disk by default and every caller downstream (features.py,
    charts.py) gets it for free without re-scanning.
    """
    cache_path = os.path.join(CACHE_DIR, f"{site_code}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    candidates = find_upstream_candidates(site_code, distance_km=distance_km)
    end = datetime.now()
    start = end - timedelta(days=check_days)

    usable = []
    for candidate in candidates[:max_candidates_checked]:
        if len(usable) >= max_sites:
            break

        has_gage_height = False
        has_discharge = False
        try:
            df = fetch_historical_series(candidate["site_code"], "00065", "_00065", start, end)
            has_gage_height = len(df) > 0
        except Exception:
            pass
        # Only spend a second round-trip checking discharge if gage height
        # wasn't available -- most usable sites report gage height, so this
        # keeps the common case to one request instead of two.
        if not has_gage_height:
            try:
                df = fetch_historical_series(candidate["site_code"], "00060", "_00060", start, end)
                has_discharge = len(df) > 0
            except Exception:
                pass

        if has_gage_height or has_discharge:
            usable.append({**candidate, "has_gage_height": has_gage_height, "has_discharge": has_discharge})

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(usable, f, indent=2)

    return usable


if __name__ == "__main__":
    import sys
    site_code = sys.argv[1] if len(sys.argv) > 1 else "01388500"
    gages = find_upstream_gages(site_code)
    for g in gages:
        print(f"{g['site_code']}: {g['name']} (gage_height={g['has_gage_height']}, discharge={g['has_discharge']})")
