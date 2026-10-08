import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run RJMCMC chains continuously until stopped, storing each chain in a campaign folder."
    )

    parser.add_argument("--python-exe", default=sys.executable, help="Python executable (default: the one running this script)")
    parser.add_argument(
        "--mcmc-script",
        default=str(Path(__file__).resolve().parent / "02_voronoi_rjmcmc.py"),
        help="Sampler script (default: 02_voronoi_rjmcmc.py next to this file)",
    )
    parser.add_argument("--csv", required=True, help="Input ray CSV")

    parser.add_argument("--chains-root", default="chains", help="Top-level folder that will contain campaign folders")
    parser.add_argument("--run-name", required=True, help="Campaign folder name, e.g. northern_europe_denmark_1y_v2_100k")

    parser.add_argument("--period", required=True, type=float)
    parser.add_argument("--min-snr", required=True, type=float)
    parser.add_argument("--min-days", required=True, type=int)
    parser.add_argument("--holdout-frac", default=0.2, type=float)
    parser.add_argument("--holdout-seed", default=17, type=int)

    parser.add_argument("--n-steps", required=True, type=int)
    parser.add_argument("--burn", required=True, type=int)
    parser.add_argument("--thin", required=True, type=int)
    parser.add_argument("--n-init", default=8, type=int)
    parser.add_argument("--seed-base", default=1000, type=int)

    parser.add_argument("--padding-km", default=30.0, type=float)
    parser.add_argument("--oversample", default=3.0, type=float)
    parser.add_argument("--grid-nx", default=140, type=int)
    parser.add_argument("--grid-ny", default=140, type=int)
    parser.add_argument("--tail-ensemble", default=60, type=int)
    parser.add_argument("--max-rays", default=None, type=int)

    parser.add_argument("--noise-std", default=None, type=float)
    parser.add_argument("--noise-frac", default=0.02, type=float)
    parser.add_argument("--vmin", default=1.2, type=float)
    parser.add_argument("--vmax", default=5.6, type=float)
    parser.add_argument("--sig-s", default=0.006, type=float)
    parser.add_argument("--sig-s2", default=0.012, type=float)
    parser.add_argument("--sig-c", default=15.0, type=float)
    parser.add_argument("--n-min", default=2, type=int)
    parser.add_argument("--n-max", default=200, type=int)

    parser.add_argument("--start-index", default=1, type=int, help="First chain number")
    parser.add_argument("--sleep-seconds", default=2.0, type=float, help="Pause between chains")
    parser.add_argument("--save-coverage", action="store_true", help="Pass --save-coverage to the sampler")

    return parser.parse_args()


REQUIRED_SUCCESS_FILES = [
    "run_summary.json",
    "posterior_grids.npz",
    "diagnostics.png",
    "posterior_summary.png",
    "prediction_comparison.png",
    "filtered_rays_train.csv",
]


def utc_now():
    return datetime.utcnow().isoformat() + "Z"


def is_chain_complete(chain_dir: Path) -> bool:
    status_file = chain_dir / "chain_status.json"
    if not status_file.exists():
        return False

    try:
        status = json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return False

    if status.get("returncode") != 0:
        return False

    for name in REQUIRED_SUCCESS_FILES:
        if not (chain_dir / name).exists():
            return False

    return True


def find_next_chain_index(campaign_dir: Path, start_index: int) -> int:
    idx = start_index
    while True:
        chain_dir = campaign_dir / f"chain_{idx:04d}"
        if not chain_dir.exists():
            return idx
        if is_chain_complete(chain_dir):
            idx += 1
            continue
        raise RuntimeError(
            f"Found existing incomplete chain folder: {chain_dir}\n"
            "Delete or move that folder before restarting this launcher."
        )


def build_command(args, chain_dir: Path, chain_idx: int):
    cmd = [
        args.python_exe,
        args.mcmc_script,
        "--csv",
        args.csv,
        "--outdir",
        str(chain_dir),
        "--period",
        str(args.period),
        "--min-snr",
        str(args.min_snr),
        "--min-days",
        str(args.min_days),
        "--holdout-frac",
        str(args.holdout_frac),
        "--holdout-seed",
        str(args.holdout_seed),
        "--n-steps",
        str(args.n_steps),
        "--burn",
        str(args.burn),
        "--thin",
        str(args.thin),
        "--n-init",
        str(args.n_init),
        "--seed",
        str(args.seed_base + chain_idx),
        "--padding-km",
        str(args.padding_km),
        "--oversample",
        str(args.oversample),
        "--grid-nx",
        str(args.grid_nx),
        "--grid-ny",
        str(args.grid_ny),
        "--tail-ensemble",
        str(args.tail_ensemble),
        "--noise-frac",
        str(args.noise_frac),
        "--vmin",
        str(args.vmin),
        "--vmax",
        str(args.vmax),
        "--sig-s",
        str(args.sig_s),
        "--sig-s2",
        str(args.sig_s2),
        "--sig-c",
        str(args.sig_c),
        "--n-min",
        str(args.n_min),
        "--n-max",
        str(args.n_max),
    ]

    if args.max_rays is not None:
        cmd.extend(["--max-rays", str(args.max_rays)])

    if args.noise_std is not None:
        cmd.extend(["--noise-std", str(args.noise_std)])

    if args.save_coverage:
        cmd.append("--save-coverage")

    return cmd


def write_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_chain_metadata(chain_dir: Path, cmd, chain_idx: int):
    meta = {
        "chain_index": chain_idx,
        "created_utc": utc_now(),
        "command": cmd,
    }
    write_json(chain_dir / "chain_meta.json", meta)

    with open(chain_dir / "command.txt", "w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(part) for part in cmd) + "\n")


def main():
    args = parse_args()

    chains_root = Path(args.chains_root)
    campaign_dir = chains_root / args.run_name
    campaign_dir.mkdir(parents=True, exist_ok=True)

    stop_file = campaign_dir / "STOP_CHAINS.txt"
    launcher_cfg = {
        "created_utc": utc_now(),
        "python_exe": args.python_exe,
        "mcmc_script": args.mcmc_script,
        "csv": args.csv,
        "period": args.period,
        "min_snr": args.min_snr,
        "min_days": args.min_days,
        "holdout_frac": args.holdout_frac,
        "holdout_seed": args.holdout_seed,
        "n_steps": args.n_steps,
        "burn": args.burn,
        "thin": args.thin,
        "n_init": args.n_init,
        "seed_base": args.seed_base,
        "padding_km": args.padding_km,
        "oversample": args.oversample,
        "grid_nx": args.grid_nx,
        "grid_ny": args.grid_ny,
        "tail_ensemble": args.tail_ensemble,
        "max_rays": args.max_rays,
        "noise_std": args.noise_std,
        "noise_frac": args.noise_frac,
        "vmin": args.vmin,
        "vmax": args.vmax,
        "sig_s": args.sig_s,
        "sig_s2": args.sig_s2,
        "sig_c": args.sig_c,
        "n_min": args.n_min,
        "n_max": args.n_max,
        "save_coverage": args.save_coverage,
        "start_index": args.start_index,
    }
    write_json(campaign_dir / "launcher_config.json", launcher_cfg)

    chain_idx = find_next_chain_index(campaign_dir, args.start_index)

    print(f"Chains root : {chains_root}")
    print(f"Campaign dir: {campaign_dir}")
    print(f"Starting at : chain_{chain_idx:04d}")
    print("Create STOP_CHAINS.txt in the campaign folder to stop after the current chain.")
    print("Ctrl+C stops immediately.\n")

    while True:
        if stop_file.exists():
            print(f"Stop file found: {stop_file}")
            print("No new chain will be started.")
            break

        chain_dir = campaign_dir / f"chain_{chain_idx:04d}"
        chain_dir.mkdir(parents=True, exist_ok=False)

        cmd = build_command(args, chain_dir, chain_idx)
        write_chain_metadata(chain_dir, cmd, chain_idx)

        print("=" * 88)
        print(f"Starting chain {chain_idx:04d}")
        print(f"Output folder: {chain_dir}")
        print("Command:")
        print(" ".join(shlex.quote(part) for part in cmd))
        print("=" * 88)

        log_path = chain_dir / "chain_output.log"
        started_utc = utc_now()
        t0 = time.time()
        write_json(
            chain_dir / "chain_status.json",
            {
                "returncode": None,
                "started_utc": started_utc,
                "finished_utc": None,
                "elapsed_seconds": None,
                "complete": False,
            },
        )

        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                result = subprocess.run(cmd, check=False, stdout=logf, stderr=subprocess.STDOUT)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            print(f"Current chain folder kept: {chain_dir}")
            print("Delete that folder before restart if the chain did not finish.")
            break

        elapsed = time.time() - t0
        missing = [name for name in REQUIRED_SUCCESS_FILES if not (chain_dir / name).exists()]
        complete = (result.returncode == 0) and (len(missing) == 0)
        write_json(
            chain_dir / "chain_status.json",
            {
                "returncode": result.returncode,
                "started_utc": started_utc,
                "finished_utc": utc_now(),
                "elapsed_seconds": elapsed,
                "complete": complete,
                "log_file": str(log_path.name),
            },
        )

        if result.returncode != 0:
            print(f"\nChain {chain_idx:04d} failed with return code {result.returncode}.")
            print(f"See folder: {chain_dir}")
            print("Stopping launcher.")
            break

        if missing:
            print(f"\nChain {chain_idx:04d} finished but expected files are missing: {missing}")
            print(f"See folder: {chain_dir}")
            print("Stopping launcher.")
            break

        print(f"\nChain {chain_idx:04d} completed in {elapsed / 60.0:.1f} min")

        chain_idx += 1

        if stop_file.exists():
            print(f"Stop file found after chain completion: {stop_file}")
            print("Stopping launcher.")
            break

        time.sleep(args.sleep_seconds)

    print("\nDone.")


if __name__ == "__main__":
    main()
