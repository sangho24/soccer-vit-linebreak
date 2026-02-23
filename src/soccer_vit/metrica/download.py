from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Guide / verify Metrica sample-data clone")
    parser.add_argument("--out", required=True, help="Output directory for sample-data clone")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"[ok] Existing directory detected: {out_dir}")
        sample_dirs = list(out_dir.rglob("Sample_Game_*"))
        if sample_dirs:
            print(f"[ok] Found sample game directories: {len(sample_dirs)}")
        else:
            print("[warn] Directory exists but Sample_Game_* folders not found yet.")
        return

    print("Network clone is not performed automatically in restricted environments.")
    print("Run the following command manually (or in a network-enabled shell):")
    print()
    print(f"  git clone https://github.com/metrica-sports/sample-data {out_dir}")
    print()
    print("Then re-run this command to validate the folder structure.")


if __name__ == "__main__":
    main()
