#!/usr/bin/env python3
"""Download Edmond dataset doi:10.17617/3.8G with resume, retries, and verification."""

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_URL = "https://edmond.mpg.de/api/datasets/:persistentId?persistentId=doi:10.17617/3.8G"
ROOT = "A_biophysical_account_of_multiplication_by_a_single_neuron"
LARGE_FILE = 100 * 1024 * 1024
MAX_RETRIES = 5
CHUNK = 4 * 1024 * 1024


def fetch_metadata():
    with urllib.request.urlopen(API_URL, timeout=120) as resp:
        return json.load(resp)


def download_file(file_id, dest_path, expected_size):
    dest_path = Path(dest_path)
    if dest_path.exists() and dest_path.stat().st_size == expected_size:
        return dest_path, "skip"

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://edmond.mpg.de/api/access/datafile/{file_id}"
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            offset = tmp_path.stat().st_size if tmp_path.exists() else 0
            if offset > expected_size:
                tmp_path.unlink()

            req = urllib.request.Request(url)
            if offset > 0:
                req.add_header("Range", f"bytes={offset}-")

            with urllib.request.urlopen(req, timeout=3600) as resp:
                status = getattr(resp, "status", resp.getcode())
                if offset > 0 and status not in (206, 200):
                    tmp_path.unlink(missing_ok=True)
                    offset = 0
                mode = "ab" if offset > 0 and status == 206 else "wb"
                if mode == "wb" and offset > 0:
                    tmp_path.unlink(missing_ok=True)
                    offset = 0

                with open(tmp_path, mode) as f:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        f.write(chunk)

            got = tmp_path.stat().st_size
            if got == expected_size:
                tmp_path.rename(dest_path)
                return dest_path, "ok"
            if got < expected_size:
                wait = min(30, 2 ** attempt)
                print(f"  incomplete {dest_path.name}: {got}/{expected_size}, retry in {wait}s", flush=True)
                time.sleep(wait)
                continue
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"size too large: got {got}, expected {expected_size}")

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            wait = min(30, 2 ** attempt)
            print(f"  attempt {attempt}/{MAX_RETRIES} failed {dest_path.name}: {exc}, retry in {wait}s", flush=True)
            time.sleep(wait)

    raise RuntimeError(f"failed after {MAX_RETRIES} attempts")


def build_jobs():
    data = fetch_metadata()
    jobs = []
    for entry in data["data"]["latestVersion"]["files"]:
        df = entry["dataFile"]
        subdir = entry.get("directoryLabel", ROOT)
        rel = Path(subdir) / entry["label"] if subdir != ROOT else Path(ROOT) / entry["label"]
        jobs.append((df["id"], BASE_DIR / rel, df["filesize"]))
    return jobs


def workers_for(size):
    return 1 if size >= LARGE_FILE else 3


def run_pass(jobs):
    pending = []
    for fid, dest, size in jobs:
        if dest.exists() and dest.stat().st_size == size:
            continue
        pending.append((fid, dest, size))

    if not pending:
        return 0, 0, 0

    total = len(pending)
    ok = skip = fail = 0
    large = [j for j in pending if j[2] >= LARGE_FILE]
    small = [j for j in pending if j[2] < LARGE_FILE]

    def run_batch(batch, workers):
        nonlocal ok, fail
        if not batch:
            return
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(download_file, fid, dest, size): (dest, size) for fid, dest, size in batch}
            for i, future in enumerate(as_completed(futures), 1):
                dest, size = futures[future]
                size_mb = size / (1024 * 1024)
                try:
                    _, status = future.result()
                    ok += 1
                    print(f"done {dest.relative_to(BASE_DIR)} ({size_mb:.1f} MB)", flush=True)
                except Exception as exc:
                    fail += 1
                    print(f"FAIL {dest.relative_to(BASE_DIR)}: {exc}", flush=True)

    print(f"Pending: {total} ({len(large)} large, {len(small)} small)", flush=True)
    run_batch(large, 1)
    run_batch(small, 3)
    return ok, skip, fail


def verify_all(jobs):
    missing = []
    bad = []
    for _, dest, size in jobs:
        if not dest.exists():
            missing.append(dest)
        elif dest.stat().st_size != size:
            bad.append((dest, dest.stat().st_size, size))
    return missing, bad


def main():
    jobs = build_jobs()
    total_size = sum(s for _, _, s in jobs)

    for round_num in range(1, 6):
        missing, bad = verify_all(jobs)
        if not missing and not bad:
            got = sum(d.stat().st_size for _, d, _ in jobs if d.exists())
            print(f"\nAll {len(jobs)} files verified ({got/1e9:.2f} / {total_size/1e9:.2f} GB)", flush=True)
            return 0

        print(f"\n=== Round {round_num}: {len(missing)} missing, {len(bad)} bad size ===", flush=True)
        ok, skip, fail = run_pass(jobs)
        print(f"Round {round_num}: {ok} downloaded, {fail} failed", flush=True)
        if fail == 0 and not missing and not bad:
            break
        time.sleep(5)

    missing, bad = verify_all(jobs)
    if missing or bad:
        print(f"\nStill incomplete: {len(missing)} missing, {len(bad)} bad", flush=True)
        for p in missing[:10]:
            print(f"  missing: {p.relative_to(BASE_DIR)}", flush=True)
        for p, g, e in bad[:10]:
            print(f"  bad: {p.relative_to(BASE_DIR)} ({g} vs {e})", flush=True)
        return 1

    print("\nAll files complete.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
