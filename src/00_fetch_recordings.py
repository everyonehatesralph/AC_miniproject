"""
00_fetch_recordings.py -- download the Group 3 recordings from Google Drive.

Source: the group's shared Google Drive folder of recordings, listed file by file in
data/raw/drive_manifest.csv (drive_file_id, drive_name, ...). That manifest is PRIVATE: it is kept out of
the public repository because the file ids give access to the participants' videos. Teammates get it from
the group's private share (docs/RUNNING.md, step 3).

Why files are saved as "<drive name>__<drive file id><ext>":
the folder contains several files with the same name (three called "Y2-057").
Saving by name alone would silently overwrite recordings. The file id keeps
every original distinct; the extension comes from the file signature.

Outputs
    data/raw/<drive_name>__<file_id><ext>   original bytes, never modified
    data/raw/download_log.csv               id, name, local file, bytes, SHA-256, container
    data/raw/recording_map.csv              local file -> participant code (parsed from
                                            the Drive name the group gave the file)

Re-running skips files already present whose SHA-256 matches the log.

Usage: .venv/Scripts/python.exe src/00_fetch_recordings.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone

import pandas as pd

from common import (DOWNLOAD_LOG, DRIVE_MANIFEST, RAW_DIR, RECORDING_MAP, REPORTS, container_ext,
                    normalise_code, rel, setup_logging, sha256_of, sniff_container)

log = setup_logging("fetch", REPORTS / "logs" / "00_fetch_recordings.log")


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


def main() -> int:
    import gdown  # imported here so the rest of the pipeline does not need it

    if not DRIVE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"{rel(DRIVE_MANIFEST)} not found. It is private (not in the public repository): copy it from the "
            "group's private share, or put the recordings in data/raw/ yourself and run with --skip-fetch "
            "(docs/RUNNING.md, step 3).")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DRIVE_MANIFEST, dtype=str, keep_default_na=False)
    if manifest["drive_file_id"].duplicated().any():
        raise ValueError("drive_manifest.csv lists a file id twice")
    previous = (pd.read_csv(DOWNLOAD_LOG, dtype=str, keep_default_na=False).set_index("drive_file_id")
                if DOWNLOAD_LOG.is_file() else pd.DataFrame())
    rows = []
    for _, item in manifest.iterrows():
        fid, name = item["drive_file_id"], item["drive_name"]
        code = normalise_code(name)
        if code is None:
            raise ValueError(f"Drive file {name!r} ({fid}) carries no participant code")
        existing = sorted(RAW_DIR.glob(f"{safe_name(name)}__{fid}.*"))
        if existing and fid in previous.index and sha256_of(existing[0]) == previous.loc[fid, "sha256"]:
            local = existing[0]
            log.info("present, hash verified: %s", local.name)
        else:
            tmp = RAW_DIR / f"{safe_name(name)}__{fid}.part"
            for attempt in range(1, 4):
                out = gdown.download(id=fid, output=str(tmp), quiet=True)
                if out and tmp.is_file() and tmp.stat().st_size > 0:
                    break
                log.warning("download attempt %d failed for %s (%s)", attempt, name, fid)
                time.sleep(3 * attempt)
            else:
                raise RuntimeError(f"could not download {name} ({fid}) after 3 attempts")
            container = sniff_container(tmp)
            if container in ("UNKNOWN", "EMPTY_OR_TRUNCATED") or container.startswith("HTML"):
                raise RuntimeError(f"{name} ({fid}) downloaded as '{container}', not media; check sharing")
            local = RAW_DIR / f"{safe_name(name)}__{fid}{container_ext(container)}"
            tmp.replace(local)
            log.info("downloaded %s -> %s (%s)", name, local.name, container)
        rows.append({
            "drive_file_id": fid, "drive_name": name, "participant_code": code,
            "local_file": rel(local), "bytes": local.stat().st_size, "sha256": sha256_of(local),
            "container": sniff_container(local), "listed_size": item["listed_size"],
            "logged_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    log_df = pd.DataFrame(rows)
    log_df.to_csv(DOWNLOAD_LOG, index=False)
    # recording_map paths are relative to data/raw/ (the audit's convention)
    pd.DataFrame({
        "original_file": [p.split("data/raw/", 1)[1] for p in log_df["local_file"]],
        "participant_code": log_df["participant_code"],
        "drive_name": log_df["drive_name"],
    }).to_csv(RECORDING_MAP, index=False)
    log.info("%d files, %d distinct participant codes, %.1f MB total", len(log_df),
             log_df["participant_code"].nunique(), log_df["bytes"].astype(int).sum() / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
