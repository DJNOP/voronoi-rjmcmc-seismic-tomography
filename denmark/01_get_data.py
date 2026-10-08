#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from itertools import combinations
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from obspy import Trace, UTCDateTime
from obspy.clients.fdsn import RoutingClient
from scipy.signal import correlate, hilbert

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, total=None, desc=None, leave=True, disable=False):
        return iterable if iterable is not None else range(total or 0)

warnings.filterwarnings("ignore")

CHANNEL_PRIORITY = ["HHZ", "BHZ", "EHZ", "SHZ", "LHZ"]
TARGET_RATE_HZ = 1.0
PRE_FILT = (0.01, 0.02, 0.40, 0.45)
DEFAULT_BBOX = {
    "minlat": 54.3,
    "maxlat": 58.2,
    "minlon": 7.2,
    "maxlon": 15.8,
}


@dataclass(frozen=True)
class StationStream:
    net: str
    sta: str
    loc: str
    cha: str
    lat: float
    lon: float
    elev_m: float
    sample_rate_hz: float


@dataclass(frozen=True)
class PeriodBand:
    period_s: float
    freqmin_hz: float
    freqmax_hz: float


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build Denmark interstation Rayleigh travel-time rays using EIDA routing"
    )
    p.add_argument("--outdir", type=Path, default=Path("denmark_data_run"))
    p.add_argument("--start", type=str, required=True)
    p.add_argument("--end", type=str, required=True, help="exclusive end date")
    p.add_argument("--network", type=str, default="DK", help="FDSN network code, default DK")
    p.add_argument("--minlat", type=float, default=DEFAULT_BBOX["minlat"])
    p.add_argument("--maxlat", type=float, default=DEFAULT_BBOX["maxlat"])
    p.add_argument("--minlon", type=float, default=DEFAULT_BBOX["minlon"])
    p.add_argument("--maxlon", type=float, default=DEFAULT_BBOX["maxlon"])
    p.add_argument("--periods", type=float, nargs="+", default=[5.0])
    p.add_argument("--fractional-bandwidth", type=float, default=0.25)
    p.add_argument("--vel-min", type=float, default=1.5)
    p.add_argument("--vel-max", type=float, default=4.5)
    p.add_argument("--min-days", type=int, default=10)
    p.add_argument("--min-snr", type=float, default=1.5)
    p.add_argument("--min-dist", type=float, default=0.0)
    p.add_argument("--max-dist", type=float, default=None)
    p.add_argument("--max-stations", type=int, default=None)
    p.add_argument("--onebit", action="store_true", default=True)
    p.add_argument("--no-onebit", dest="onebit", action="store_false")
    p.add_argument("--cache-dir", type=Path, default=Path("denmark_trace_cache"))
    p.add_argument("--clear-cache", action="store_true")
    p.add_argument("--save-plots", action="store_true", help="save per-pair CCF plots")
    p.add_argument("--show-day-bars", action="store_true")
    p.add_argument("--label-stations", action="store_true", default=True)
    p.add_argument("--no-label-stations", dest="label_stations", action="store_false")
    return p


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def normalize_location(loc: Optional[str]) -> str:
    if loc is None:
        return ""
    loc = str(loc).strip()
    if loc.lower() in {"nan", "none"}:
        return ""
    return loc


def period_bands(periods: Iterable[float], fractional_bandwidth: float) -> List[PeriodBand]:
    out: List[PeriodBand] = []
    for p in sorted({float(v) for v in periods}):
        if p <= 0:
            raise ValueError("All periods must be positive")
        f0 = 1.0 / p
        fmin = max(1e-4, f0 * (1.0 - fractional_bandwidth))
        fmax = f0 * (1.0 + fractional_bandwidth)
        out.append(PeriodBand(period_s=p, freqmin_hz=fmin, freqmax_hz=fmax))
    return out


def build_days(start: UTCDateTime, end: UTCDateTime) -> List[UTCDateTime]:
    days: List[UTCDateTime] = []
    t = start
    while t < end:
        days.append(t)
        t += 24 * 3600
    return days


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2.0) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2.0) ** 2
    return 2.0 * R * atan2(sqrt(a), sqrt(1.0 - a))


def is_no_data_exception(exc: Exception) -> bool:
    txt = str(exc)
    return ("No data available for request" in txt) or ("HTTP Status code: 204" in txt)


def cache_key_base(station: StationStream, day_start: UTCDateTime) -> str:
    loc = station.loc if station.loc else "--"
    return f"{station.net}.{station.sta}.{loc}.{station.cha}.{day_start.strftime('%Y%m%d')}"


# ------------------------------------------------------------
# Station discovery
# ------------------------------------------------------------

def _channel_rank(code: str) -> Tuple[int, int]:
    code = str(code).strip().upper()
    try:
        idx = CHANNEL_PRIORITY.index(code)
    except ValueError:
        idx = len(CHANNEL_PRIORITY)
    # second component prefers broadband-ish channels lexically earlier less important
    return idx, 0


def discover_station_streams(
    client: RoutingClient,
    network: str,
    start: UTCDateTime,
    end: UTCDateTime,
    minlat: float,
    maxlat: float,
    minlon: float,
    maxlon: float,
    max_stations: Optional[int],
) -> Dict[Tuple[str, str], StationStream]:
    print("Discovering Denmark stations via EIDA routing...")
    inv = client.get_stations(
        network=network,
        channel="*HZ",
        starttime=start,
        endtime=end,
        minlatitude=minlat,
        maxlatitude=maxlat,
        minlongitude=minlon,
        maxlongitude=maxlon,
        level="channel",
    )

    candidates: Dict[Tuple[str, str], List[StationStream]] = {}
    for net in inv:
        for sta in net:
            station_key = (net.code, sta.code)
            items: List[StationStream] = candidates.setdefault(station_key, [])
            for cha in sta.channels:
                code = str(cha.code).upper()
                if not code.endswith("Z"):
                    continue
                items.append(
                    StationStream(
                        net=net.code,
                        sta=sta.code,
                        loc=normalize_location(cha.location_code),
                        cha=code,
                        lat=float(cha.latitude),
                        lon=float(cha.longitude),
                        elev_m=float(getattr(cha, "elevation", sta.elevation or 0.0)),
                        sample_rate_hz=float(getattr(cha, "sample_rate", 0.0) or 0.0),
                    )
                )

    chosen: Dict[Tuple[str, str], StationStream] = {}
    for key, items in candidates.items():
        if not items:
            continue
        items_sorted = sorted(
            items,
            key=lambda s: (
                _channel_rank(s.cha)[0],
                -float(s.sample_rate_hz),
                0 if s.loc == "" else 1,
                s.loc,
            ),
        )
        chosen[key] = items_sorted[0]

    selected = sorted(chosen.values(), key=lambda s: (s.net, s.sta))
    if max_stations is not None and len(selected) > max_stations:
        selected = selected[:max_stations]

    out = {(s.net, s.sta): s for s in selected}
    for s in selected:
        loc_disp = s.loc if s.loc else "--"
        print(
            f"  {s.net}.{s.sta}.{loc_disp}.{s.cha}  lat={s.lat:.4f}, lon={s.lon:.4f}, fs={s.sample_rate_hz:g} Hz"
        )
    return out


# ------------------------------------------------------------
# Waveform handling
# ------------------------------------------------------------

def get_station_response(client: RoutingClient, station: StationStream, start: UTCDateTime, end: UTCDateTime):
    loc = station.loc if station.loc else "*"
    return client.get_stations(
        network=station.net,
        station=station.sta,
        location=loc,
        channel=station.cha,
        starttime=start,
        endtime=end,
        level="response",
    )


def homogenize_stream_sampling_rates(st, target_rate: float = TARGET_RATE_HZ, tol: float = 1e-3):
    for tr in st:
        sr = float(tr.stats.sampling_rate)
        if abs(sr - target_rate) <= tol:
            tr.stats.sampling_rate = target_rate
        else:
            tr.interpolate(sampling_rate=target_rate, method="lanczos", a=8)
    return st


def fetch_or_build_base_trace(
    client: RoutingClient,
    station: StationStream,
    day_start: UTCDateTime,
    day_end: UTCDateTime,
    cache_dir: Path,
) -> Optional[np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key_base(station, day_start)
    ok_file = cache_dir / f"{key}.npz"
    no_file = cache_dir / f"{key}.nodata"

    if ok_file.exists():
        return np.load(ok_file)["data"]
    if no_file.exists():
        return None

    loc_query = station.loc if station.loc else "*"
    try:
        st = client.get_waveforms(
            network=station.net,
            station=station.sta,
            location=loc_query,
            channel=station.cha,
            starttime=day_start,
            endtime=day_end,
        )
    except Exception as e:
        if is_no_data_exception(e):
            no_file.write_text("nodata\n", encoding="utf-8")
            return None
        raise

    if len(st) == 0:
        no_file.write_text("nodata\n", encoding="utf-8")
        return None

    st = st.select(channel=station.cha)
    if len(st) == 0:
        no_file.write_text("nodata\n", encoding="utf-8")
        return None

    try:
        st = homogenize_stream_sampling_rates(st, target_rate=TARGET_RATE_HZ, tol=1e-3)
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        st = st.sort(keys=["npts"], reverse=True)
        st = st[:1]

    if len(st) > 1:
        st = st.sort(keys=["npts"], reverse=True)
        st = st[:1]

    tr = st[0].copy()
    tr.trim(day_start, day_end, pad=True, fill_value=0)
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05, type="cosine")

    try:
        inv = get_station_response(client, station, day_start, day_end)
        tr.remove_response(inventory=inv, output="VEL", pre_filt=PRE_FILT, water_level=60)
    except Exception as e:
        if is_no_data_exception(e):
            no_file.write_text("nodata\n", encoding="utf-8")
            return None
        raise

    if abs(float(tr.stats.sampling_rate) - TARGET_RATE_HZ) > 1e-6:
        tr.resample(TARGET_RATE_HZ)

    data = np.asarray(tr.data, dtype=np.float32)
    if np.allclose(data, 0.0):
        no_file.write_text("nodata\n", encoding="utf-8")
        return None

    expected_n = int((day_end - day_start) * TARGET_RATE_HZ)
    if len(data) != expected_n:
        if len(data) < expected_n:
            data = np.pad(data, (0, expected_n - len(data)))
        else:
            data = data[:expected_n]

    np.savez_compressed(ok_file, data=data)
    return data


def filter_trace(base_data: np.ndarray, freqmin: float, freqmax: float, onebit: bool) -> Optional[np.ndarray]:
    tr = Trace(data=np.asarray(base_data, dtype=np.float64).copy())
    tr.stats.sampling_rate = TARGET_RATE_HZ
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    data = np.asarray(tr.data, dtype=np.float64)
    if np.allclose(data, 0.0):
        return None
    if onebit:
        data = np.sign(data)
    return data


# ------------------------------------------------------------
# CCF / picking
# ------------------------------------------------------------

def compute_ccf(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ccf = correlate(x, y, mode="full", method="fft").astype(np.float64)
    ccf /= len(x)
    return ccf


def pick_traveltime_from_stack(ccf_stack: np.ndarray, dist_km: float, vel_min: float, vel_max: float):
    n = (len(ccf_stack) + 1) // 2
    lags = np.arange(-n + 1, n) / TARGET_RATE_HZ
    pos_mask = lags >= 0
    lags_pos = lags[pos_mask]
    ccf_pos = ccf_stack[pos_mask]
    env = np.abs(hilbert(ccf_pos))

    tmin = dist_km / vel_max
    tmax = dist_km / vel_min
    win_mask = (lags_pos >= tmin) & (lags_pos <= tmax)
    if not np.any(win_mask):
        return None

    idx_local = np.argmax(env[win_mask])
    idx_global = np.where(win_mask)[0][0] + idx_local
    peak_time_s = float(lags_pos[idx_global])
    peak_amp = float(env[idx_global])

    noise_mask = ~win_mask
    noise_level = float(np.median(env[noise_mask])) if np.any(noise_mask) else float(np.median(env))
    snr = np.inf if noise_level == 0 else peak_amp / noise_level
    velocity_km_s = dist_km / peak_time_s if peak_time_s > 0 else np.nan

    return {
        "lags_pos": lags_pos,
        "ccf_pos": ccf_pos,
        "env_pos": env,
        "tmin": tmin,
        "tmax": tmax,
        "traveltime_s": peak_time_s,
        "velocity_km_s": velocity_km_s,
        "snr": snr,
    }


def save_ccf_plot(result: dict, pair_name: str, period_s: float, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(result["lags_pos"], result["ccf_pos"], label="Positive-lag stacked CCF")
    plt.plot(result["lags_pos"], result["env_pos"], label="Envelope")
    plt.axvspan(result["tmin"], result["tmax"], alpha=0.2, label="Search window")
    plt.axvline(result["traveltime_s"], linestyle="--", label=f"Pick = {result['traveltime_s']:.1f} s")
    plt.xlabel("Lag time [s]")
    plt.ylabel("Amplitude")
    plt.title(f"{pair_name} | {period_s:g} s")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# ------------------------------------------------------------
# Geographic map (standard output)
# ------------------------------------------------------------

def save_geographic_ray_map(
    rays_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    outpath: Path,
    title: str,
    label_stations: bool,
) -> None:
    minlon = float(min(stations_df["lon"].min(), rays_df[["lon1", "lon2"]].min().min()) - 1.0)
    maxlon = float(max(stations_df["lon"].max(), rays_df[["lon1", "lon2"]].max().max()) + 1.0)
    minlat = float(min(stations_df["lat"].min(), rays_df[["lat1", "lat2"]].min().min()) - 0.8)
    maxlat = float(max(stations_df["lat"].max(), rays_df[["lat1", "lat2"]].max().max()) + 0.8)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        proj = ccrs.LambertConformal(central_longitude=10.5, central_latitude=56.0)
        pc = ccrs.PlateCarree()

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=proj)
        ax.set_extent([minlon, maxlon, minlat, maxlat], crs=pc)
        ax.add_feature(cfeature.LAND, facecolor="#f0f0f0")
        ax.add_feature(cfeature.OCEAN, facecolor="#dfefff")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.6)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")

        for row in rays_df.itertuples(index=False):
            ax.plot([row.lon1, row.lon2], [row.lat1, row.lat2], transform=pc, color="tab:blue", alpha=0.35, lw=0.9)

        ax.scatter(stations_df["lon"], stations_df["lat"], transform=pc, s=35, color="crimson", edgecolor="black", zorder=5)
        if label_stations:
            for row in stations_df.itertuples(index=False):
                ax.text(row.lon + 0.03, row.lat + 0.03, row.sta, transform=pc, fontsize=8, zorder=6)
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(outpath, dpi=170)
        plt.close(fig)
        return
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(10, 10))
    for row in rays_df.itertuples(index=False):
        ax.plot([row.lon1, row.lon2], [row.lat1, row.lat2], color="tab:blue", alpha=0.35, lw=0.9)
    ax.scatter(stations_df["lon"], stations_df["lat"], s=35, color="crimson", edgecolor="black", zorder=5)
    if label_stations:
        for row in stations_df.itertuples(index=False):
            ax.text(row.lon + 0.03, row.lat + 0.03, row.sta, fontsize=8)
    ax.set_xlim(minlon, maxlon)
    ax.set_ylim(minlat, maxlat)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title + " (no cartopy)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(outpath, dpi=170)
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    args = make_parser().parse_args()

    outdir = args.outdir
    csv_dir = outdir / "csv"
    plot_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if args.clear_cache and args.cache_dir.exists():
        for f in args.cache_dir.glob("*"):
            f.unlink()

    start = UTCDateTime(f"{args.start}T00:00:00")
    end = UTCDateTime(f"{args.end}T00:00:00")
    all_days = build_days(start, end)
    bands = period_bands(args.periods, args.fractional_bandwidth)

    client = RoutingClient("eida-routing")
    stations = discover_station_streams(
        client=client,
        network=args.network,
        start=start,
        end=end,
        minlat=args.minlat,
        maxlat=args.maxlat,
        minlon=args.minlon,
        maxlon=args.maxlon,
        max_stations=args.max_stations,
    )
    if len(stations) < 2:
        raise RuntimeError("Fewer than 2 Denmark stations available after discovery")

    stations_df = pd.DataFrame([
        {
            "net": s.net,
            "sta": s.sta,
            "loc": s.loc,
            "cha": s.cha,
            "lat": s.lat,
            "lon": s.lon,
            "elev_m": s.elev_m,
            "sample_rate_hz": s.sample_rate_hz,
        }
        for s in stations.values()
    ]).sort_values(["net", "sta"]).reset_index(drop=True)
    stations_df.to_csv(csv_dir / "stations_used.csv", index=False)

    station_keys = sorted(stations.keys())
    pairs = list(combinations(station_keys, 2))
    n_tasks = len(pairs) * len(bands)
    print(f"\nTrying {len(pairs)} station pairs over {len(all_days)} daily windows and {len(bands)} periods")

    pair_rows_all: List[dict] = []
    pair_rows_kept: List[dict] = []
    filtered_cache: Dict[Tuple[str, float], Optional[np.ndarray]] = {}

    outer = tqdm(pairs, desc="Pairs", total=len(pairs))
    for k1, k2 in outer:
        s1 = stations[k1]
        s2 = stations[k2]
        pair_name = f"{s1.net}.{s1.sta}__{s2.net}.{s2.sta}"
        outer.set_postfix_str(pair_name)
        dist_km = haversine_km(s1.lat, s1.lon, s2.lat, s2.lon)
        if dist_km < args.min_dist:
            continue
        if args.max_dist is not None and dist_km > args.max_dist:
            continue

        stacks = {b.period_s: None for b in bands}
        used = {b.period_s: 0 for b in bands}
        missing_1 = 0
        missing_2 = 0
        missing_both = 0

        day_iter = tqdm(all_days, desc=pair_name, leave=False, disable=not args.show_day_bars)
        for day_start in day_iter:
            day_end = day_start + 24 * 3600
            x_base = fetch_or_build_base_trace(client, s1, day_start, day_end, args.cache_dir)
            y_base = fetch_or_build_base_trace(client, s2, day_start, day_end, args.cache_dir)

            if x_base is None and y_base is None:
                missing_both += 1
                continue
            if x_base is None:
                missing_1 += 1
                continue
            if y_base is None:
                missing_2 += 1
                continue
            if len(x_base) != len(y_base):
                continue

            base_key_1 = cache_key_base(s1, day_start)
            base_key_2 = cache_key_base(s2, day_start)
            for band in bands:
                fk1 = (base_key_1, band.period_s)
                fk2 = (base_key_2, band.period_s)
                if fk1 not in filtered_cache:
                    filtered_cache[fk1] = filter_trace(x_base, band.freqmin_hz, band.freqmax_hz, args.onebit)
                if fk2 not in filtered_cache:
                    filtered_cache[fk2] = filter_trace(y_base, band.freqmin_hz, band.freqmax_hz, args.onebit)
                x = filtered_cache[fk1]
                y = filtered_cache[fk2]
                if x is None or y is None or len(x) != len(y):
                    continue
                ccf = compute_ccf(x, y)
                stacks[band.period_s] = ccf if stacks[band.period_s] is None else (stacks[band.period_s] + ccf)
                used[band.period_s] += 1

        for band in bands:
            row = {
                "net1": s1.net,
                "sta1": s1.sta,
                "loc1": s1.loc,
                "cha1": s1.cha,
                "lat1": s1.lat,
                "lon1": s1.lon,
                "net2": s2.net,
                "sta2": s2.sta,
                "loc2": s2.loc,
                "cha2": s2.cha,
                "lat2": s2.lat,
                "lon2": s2.lon,
                "dist_km": dist_km,
                "period_s": band.period_s,
                "freqmin_hz": band.freqmin_hz,
                "freqmax_hz": band.freqmax_hz,
                "days_used": used[band.period_s],
                "days_missing_sta1": missing_1,
                "days_missing_sta2": missing_2,
                "days_missing_both": missing_both,
            }

            if stacks[band.period_s] is None or used[band.period_s] == 0:
                row.update({"status": "no_overlap", "traveltime_s": np.nan, "velocity_km_s": np.nan, "snr": np.nan})
                pair_rows_all.append(row)
                continue

            ccf_stack = stacks[band.period_s] / used[band.period_s]
            pick = pick_traveltime_from_stack(ccf_stack, dist_km, args.vel_min, args.vel_max)
            if pick is None:
                row.update({"status": "no_pick", "traveltime_s": np.nan, "velocity_km_s": np.nan, "snr": np.nan})
                pair_rows_all.append(row)
                continue

            status = "kept" if (used[band.period_s] >= args.min_days and pick["snr"] >= args.min_snr) else "rejected"
            row.update({
                "status": status,
                "traveltime_s": pick["traveltime_s"],
                "velocity_km_s": pick["velocity_km_s"],
                "snr": pick["snr"],
            })
            pair_rows_all.append(row)

            summary = (
                f"days_used={used[band.period_s]}, missing1={missing_1}, missing2={missing_2}, both={missing_both} | "
                f"P={band.period_s:g}s tt={pick['traveltime_s']:.1f}s vel={pick['velocity_km_s']:.3f} km/s snr={pick['snr']:.2f} | {status}"
            )
            print(f"\n=== Pair {pair_name} ===\n  {summary}")

            if status == "kept":
                pair_rows_kept.append(row.copy())
                if args.save_plots:
                    plot_name = f"{pair_name}__P{band.period_s:g}.png".replace(".", "p")
                    save_ccf_plot(pick, pair_name, band.period_s, plot_dir / plot_name)

    df_all = pd.DataFrame(pair_rows_all).sort_values(["net1", "sta1", "net2", "sta2", "period_s"]).reset_index(drop=True)
    df_keep = pd.DataFrame(pair_rows_kept).sort_values(["net1", "sta1", "net2", "sta2", "period_s"]).reset_index(drop=True)

    df_all.to_csv(csv_dir / "pair_attempts_all.csv", index=False)
    df_keep.to_csv(csv_dir / "ray_data_denmark.csv", index=False)

    if len(df_keep) > 0:
        save_geographic_ray_map(
            rays_df=df_keep,
            stations_df=stations_df,
            outpath=outdir / "denmark_rays_map.png",
            title=f"Denmark station map and used rays ({len(df_keep)} rays)",
            label_stations=args.label_stations,
        )

    summary = {
        "network": args.network,
        "n_stations": int(len(stations)),
        "n_candidate_pairs": int(len(pairs)),
        "n_candidate_tasks": int(n_tasks),
        "n_attempted_rows": int(len(df_all)),
        "n_kept_rays": int(len(df_keep)),
        "start": args.start,
        "end_exclusive": args.end,
        "periods_s": [b.period_s for b in bands],
        "fractional_bandwidth": float(args.fractional_bandwidth),
        "min_days": int(args.min_days),
        "min_snr": float(args.min_snr),
        "onebit": bool(args.onebit),
        "cache_dir": str(args.cache_dir),
        "bbox": {"minlat": args.minlat, "maxlat": args.maxlat, "minlon": args.minlon, "maxlon": args.maxlon},
    }
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(f"  {csv_dir / 'stations_used.csv'}")
    print(f"  {csv_dir / 'pair_attempts_all.csv'}")
    print(f"  {csv_dir / 'ray_data_denmark.csv'}")
    print(f"  {outdir / 'denmark_rays_map.png'}")
    print(f"  {outdir / 'run_summary.json'}")
    if args.save_plots:
        print(f"  plots in {plot_dir}")

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()